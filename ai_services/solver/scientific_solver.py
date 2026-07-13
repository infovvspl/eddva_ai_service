import os
import io
import base64
import json
import ast
import logging
import traceback
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy import integrate, optimize, stats
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import pint
import pubchempy as pcp
import httpx
import subprocess
from typing import Any, Dict, Optional, List
from ai_services.core.llm_client import get_llm
from ai_services.solver.formula_retriever import formula_retriever

try:
    from openbabel import openbabel as ob
except ImportError:
    ob = None

class RuleBasedChiralDetector:
    """Helper to detect and explain chiral centers using RDKit."""
    @staticmethod
    def detect(mol: Any) -> List[Dict[str, Any]]:
        if not mol: return []
        # Ensure hydrogens are present for accurate chirality detection
        mol_with_hs = Chem.AddHs(mol)
        AllChem.AssignStereochemistry(mol_with_hs, force=True, cleanIt=True)
        centers = Chem.FindMolChiralCenters(mol_with_hs, includeUnassigned=True)
        
        results = []
        for idx, label in centers:
            atom = mol_with_hs.GetAtomWithIdx(idx)
            neighbors = []
            for n in atom.GetNeighbors():
                neighbors.append({
                    "symbol": n.GetSymbol(),
                    "atomic_num": n.GetAtomicNum(),
                    "degree": n.GetDegree()
                })
            
            results.append({
                "atom_index": idx,
                "label": label, # 'R', 'S', or '?'
                "atom_symbol": atom.GetSymbol(),
                "neighbors": neighbors,
                "explanation": f"Atom {atom.GetSymbol()} at index {idx} is a chiral center with {len(neighbors)} attachments."
            })
        return results

class StepValidator:
    """Helper to verify mathematical equivalence between steps using SymPy."""
    @staticmethod
    def verify_substitution(original, candidate):
        """Mathematically verify a substitution using SymPy simplification."""
        try:
            # Assume original and candidate are SymPy expressions or strings
            diff = sp.simplify(original - candidate)
            is_valid = (diff == 0)
            return {"valid": is_valid, "diff": str(diff)}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    @staticmethod
    def validate_steps(steps: List[str]) -> List[Dict[str, Any]]:
        results = []
        parsed_steps = []
        
        for step in steps:
            try:
                # Basic LaTeX to SymPy conversion (very simplified)
                clean_step = step.replace("$", "").replace("\\", "").strip()
                # If it's an equation, split it
                if "=" in clean_step:
                    lhs, rhs = clean_step.split("=", 1)
                    parsed = sp.simplify(f"({lhs}) - ({rhs})")
                else:
                    parsed = sp.simplify(clean_step)
                parsed_steps.append(parsed)
            except Exception as e:
                parsed_steps.append(None)
        
        for i in range(1, len(parsed_steps)):
            prev = parsed_steps[i-1]
            curr = parsed_steps[i]
            if prev is not None and curr is not None:
                # Check if curr - prev simplifies to 0 (equivalence)
                # Or if it's an equation, check if they represent the same set of solutions (harder, so we check expression difference)
                is_valid = sp.simplify(curr - prev) == 0
                results.append({
                    "from_step": i-1,
                    "to_step": i,
                    "is_valid": is_valid,
                    "note": "Equivalent" if is_valid else "Potential logical jump or error"
                })
            else:
                results.append({
                    "from_step": i-1,
                    "to_step": i,
                    "is_valid": False,
                    "note": "Could not parse steps for verification"
                })
        return results

class OpenBabelFallback:
    """Wrapper for Open Babel functionality with fallback to subprocess CLI."""
    @staticmethod
    def convert(input_data: str, in_fmt: str, out_fmt: str) -> str:
        if ob:
            obmol = ob.OBMol()
            obconversion = ob.OBConversion()
            obconversion.SetInAndOutFormats(in_fmt, out_fmt)
            obconversion.ReadString(obmol, input_data)
            return obconversion.WriteString(obmol)
        else:
            # Try subprocess
            try:
                res = subprocess.run(
                    ["obabel", f"-i{in_fmt}", f"-o{out_fmt}"],
                    input=input_data.encode(),
                    capture_output=True,
                    check=True
                )
                return res.stdout.decode()
            except Exception:
                return "Open Babel not available (neither library nor CLI)."

class MaximaFallback:
    """Wrapper for Maxima functionality via subprocess."""
    @staticmethod
    def evaluate(expression: str) -> str:
        try:
            # Simple maxima call
            cmd = f"print(expand({expression}));"
            res = subprocess.run(
                ["maxima", "--very-quiet", "-r", cmd],
                capture_output=True,
                check=True
            )
            return res.stdout.decode().strip()
        except Exception:
            return "Maxima not available via CLI."

logger = logging.getLogger("ai_services.scientific_solver")
ureg = pint.UnitRegistry()

class ScientificSolver:
    def __init__(self):
        self.llm = get_llm()
        self.allowed_modules = {
            "np": np,
            "sp": sp,
            "plt": plt,
            "integrate": integrate,
            "optimize": optimize,
            "stats": stats,
            "Chem": Chem,
            "Descriptors": Descriptors,
            "AllChem": AllChem,
            "pcp": pcp,
            "ureg": ureg,
            "ChiralDetector": RuleBasedChiralDetector,
            "StepValidator": StepValidator,
            "OpenBabel": OpenBabelFallback,
            "Maxima": MaximaFallback,
        }

    def _clean_generated_code(self, content: str) -> str:
        """Extract executable Python from an LLM response."""
        code = (content or "").strip()
        if not code:
            return ""

        if "```python" in code:
            code = code.split("```python", 1)[1].split("```", 1)[0].strip()
        elif "```" in code:
            code = code.split("```", 1)[1].split("```", 1)[0].strip()

        lines = code.splitlines()
        start_markers = (
            "import ",
            "from ",
            "x =",
            "y =",
            "t =",
            "n =",
            "a =",
            "b =",
            "c =",
            "f =",
            "g =",
            "FINAL_RESULT",
            "def ",
            "#",
        )

        start_idx = 0
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(start_markers):
                start_idx = idx
                break
        code = "\n".join(lines[start_idx:]).strip()

        # Drop common non-code tails if a model ignored the instruction.
        tail_markers = ("Explanation:", "Answer:", "Final answer:", "Steps:")
        cleaned_lines = []
        for line in code.splitlines():
            if line.strip().startswith(tail_markers):
                break
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    def _code_is_valid_python(self, code: str) -> Optional[str]:
        try:
            ast.parse(code)
            return None
        except SyntaxError as exc:
            return f"{exc.msg} (line {exc.lineno})"

    async def fetch_nist_data(self, name: str) -> str:
        """Helper to fetch basic data from NIST WebBook (simulated/basic parsing)."""
        url = f"https://webbook.nist.gov/cgi/cbook.cgi?Name={name}&Units=SI"
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    # Very basic: just return first 1000 chars of HTML for the LLM to parse or mention
                    return f"NIST data for {name} found. Summary: {resp.text[:1000]}..."
                return f"No NIST data found for {name}."
            except Exception as e:
                return f"Error fetching NIST data: {str(e)}"

    def _execute_code(self, code: str) -> Dict[str, Any]:
        """
        Execute LLM-generated Python in an ISOLATED subprocess (security boundary
        — see docs/PRODUCTION_AUDIT.md C1). The child gets a hard wall-clock
        timeout, a secret-stripped environment (no API keys / DB creds reachable),
        and (on POSIX) CPU/memory/file-size rlimits. Any failure degrades to the
        LLM path in solve(), exactly as before.

        Disable entirely with SOLVER_EXEC_ENABLED=false.
        """
        import os
        import sys
        import subprocess

        if os.getenv("SOLVER_EXEC_ENABLED", "true").lower() not in ("true", "1", "yes"):
            return {"success": False, "stdout": "", "results": {}, "graphs": [],
                    "error": "scientific solver code execution is disabled (SOLVER_EXEC_ENABLED=false)"}

        timeout_s = int(os.getenv("SOLVER_EXEC_TIMEOUT", "15"))
        # project root = .../ai_services/solver/scientific_solver.py -> up 3
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Minimal, secret-free environment. Deliberately excludes GROQ_*, DB_*,
        # REDIS_URL, SARVAM/GEMINI keys, etc. so exec'd code cannot exfiltrate them.
        clean_env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": base_dir,
            "MPLBACKEND": "Agg",
            "HOME": os.environ.get("HOME") or os.environ.get("TEMP") or "/tmp",
        }
        for _k in ("SYSTEMROOT", "TEMP", "TMP", "LD_LIBRARY_PATH"):
            if os.environ.get(_k):
                clean_env[_k] = os.environ[_k]

        preexec = None
        if os.name == "posix":
            def _apply_limits():
                import resource
                cpu = timeout_s + 2
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
                mem = 2 * 1024 * 1024 * 1024  # 2 GB address space
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
                fsize = 50 * 1024 * 1024      # 50 MB max file write
                resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
                os.setsid()  # own process group so timeout kill takes children too
            preexec = _apply_limits

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "ai_services.solver._sandbox_runner"],
                input=json.dumps({"code": code}),
                capture_output=True, text=True,
                timeout=timeout_s, cwd=base_dir, env=clean_env,
                preexec_fn=preexec,
            )
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "results": {}, "graphs": [],
                    "error": f"execution timed out after {timeout_s}s"}
        except Exception as e:
            return {"success": False, "stdout": "", "results": {}, "graphs": [],
                    "error": f"sandbox launch failed: {e}"}

        from ai_services.solver._sandbox_runner import RESULT_SENTINEL
        out = proc.stdout or ""
        idx = out.rfind(RESULT_SENTINEL)
        if idx == -1:
            return {"success": False, "stdout": out, "results": {}, "graphs": [],
                    "error": f"sandbox produced no result (rc={proc.returncode}): {(proc.stderr or '')[:500]}"}
        try:
            return json.loads(out[idx + len(RESULT_SENTINEL):])
        except Exception as e:
            return {"success": False, "stdout": out, "results": {}, "graphs": [],
                    "error": f"could not parse sandbox result: {e}"}

    async def solve(self, question: str, mode: str = "detailed", vertical: str = "coaching") -> Dict[str, Any]:
        """Main entry point: Generate code -> Execute -> Synthesize answer."""

        # Step 1.1: Retrieve relevant formulas from the Knowledge Base (PDFs).
        #
        # The KB is built from JEE/NEET formula sheets (see data/knowledge_base/:
        # IIT-JEE-Formula-PCM, NEET/JEE maths formulae, ...). Grounding a Class 1-10
        # answer with IIT-JEE formulae would push it above grade level, so the KB is
        # used for the coaching vertical only. School answers are solved from first
        # principles at board level instead.
        is_school = (vertical or "").lower() == "school"
        kb_formulas = [] if is_school else formula_retriever.retrieve(question)
        formula_context = ""
        if kb_formulas:
            formula_context = "\n\nVERIFIED FORMULAS FROM KNOWLEDGE BASE:\n"
            for f in kb_formulas:
                formula_context += f"- {f['text']} (Source: {f['source']})\n"

        # Step 1.2: Generate Solution Code
        audience = (
            "a school tutor app for Classes 1-10 (CBSE/ICSE/State board). Keep the "
            "method at board level — no competitive-exam shortcuts or advanced tricks"
            if is_school else
            "a JEE/NEET/CBSE tutor app"
        )
        system_prompt = (
            f"You are a Scientific Code Generator for {audience}. "
            "Your goal is to write Python code that solves a scientific or mathematical question with 100% accuracy.\n\n"
            "PRE-IMPORTED MODULES (already available, do NOT import them):\n"
            "- np: NumPy\n"
            "- sp: SymPy\n"
            "- plt: Matplotlib.pyplot\n"
            "- integrate, optimize, stats: SciPy modules\n"
            "- Chem, Descriptors, AllChem: RDKit modules\n"
            "- pcp: PubChemPy\n"
            "- ureg: Pint UnitRegistry\n\n"
            "RULES:\n"
            "1. Focus on accuracy. Use SymPy (sp) for complex derivations and equation solving.\n"
            "2. Use RDKit (Chem, AllChem) for advanced chemistry. Use ChiralDetector.detect(mol) for rule-based chirality analysis.\n"
            "3. Use StepValidator.validate_steps([step1, step2, ...]) to verify mathematical logic if helpful.\n"
            "4. Use OpenBabel.convert(data, in_fmt, out_fmt) or Maxima.evaluate(expr) as fallbacks if primary tools (RDKit/SymPy) are insufficient.\n"
            "5. If a plot is helpful, use plt.plot() etc. Do NOT use plt.show().\n"
            "6. FINAL_RESULT VARIABLE: You MUST store the ultimate final numerical or symbolic answer in a variable named 'FINAL_RESULT'. Solve until the LAST target is reached.\n"
            "7. CALCULUS RIGOR (CRITICAL):\n"
            "   - NEVER change powers while substituting (e.g., $(x-t)^5$ stays power 5).\n"
            "   - NEVER invent new factors or coefficients.\n"
            "   - ALWAYS derive $g'(x)$ using FTC/Leibniz Rule.\n"
            "   - SOLVE $g'(x)=0$ for extrema, NOT $g(x)=0$.\n"
            "8. SUBSTITUTION VALIDATOR: Use 'StepValidator.verify_substitution(original, candidate)' to prove your algebra is correct before moving to the next step.\n"
            "9. Respond ONLY with valid Python code. No markdown fences. No explanation.\n\n"
            "SCIENTIFIC TRAP DETECTION (MANDATORY CHECKS):\n"
            "1. INTEGRAL EQUATIONS: Check for constant solutions (f(t)=k). Verify domains/singularities (denominators ≠ 0). Enforce consistency of integral constants (C = ∫f(t)dt).\n"
            "2. EXTREMA TRAPS: Use Leibniz Rule for differentiation under the integral sign—DO NOT brute force integrate if (x-t) is present. Perform sign analysis on odd vs even powers.\n"
            "3. VECTOR LOGIC: Distinguish Scaling vs Rotation vs Reflection. v₁·v₂=0 means 90° rotation.\n"
            "4. CHEMISTRY CHIRALITY: Check for symmetry (identical groups = NOT chiral). Check side-chains and rings for hidden centers.\n"
            "5. PHYSICS CONCEPTS: a=-kx+c is shifted SHM, not simple SHM. If acceleration is variable (a=kt), DO NOT use constant acceleration (SUVAT) formulas.\n"
            "6. DOMAIN VALIDITY: Check for extraneous roots in radicals (√(x-1)=x-3) and enforce log domains (ln(u) → u>0).\n\n"
            f"{formula_context}"
        )
        
        user_prompt = f"Question: {question}\n\nWrite the Python code to solve this."
        
        llm_resp = self.llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="llama-3.3-70b-versatile",
            json_mode=False
        )
        
        code = self._clean_generated_code(llm_resp["content"])
        syntax_error = self._code_is_valid_python(code)

        if syntax_error:
            # The model sometimes wraps the code in JSON/prose, or just emits
            # malformed Python (observed in prod: "unmatched '}'"). One bad
            # generation shouldn't kill the solver — retry once, deterministically,
            # with an explicit "raw Python only" instruction. Log a snippet so the
            # failure is diagnosable instead of silent.
            logger.warning(
                "Scientific code generation produced invalid Python (%s) — retrying once. Snippet: %r",
                syntax_error, code[:200],
            )
            retry_resp = self.llm.complete(
                system_prompt=system_prompt + (
                    "\n\nCRITICAL OUTPUT RULE: Respond with ONLY raw, executable Python source code. "
                    "No JSON. No markdown code fences. No prose, headings, or commentary. "
                    "The very first character of your response must be the first character of the code."
                ),
                user_prompt=user_prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.0,
                json_mode=False,
            )
            code = self._clean_generated_code(retry_resp["content"])
            syntax_error = self._code_is_valid_python(code)

        if syntax_error:
            logger.warning(
                "Scientific code generation still invalid after retry: %s | snippet: %r",
                syntax_error, code[:200],
            )
            return {
                "success": False,
                "error": f"Generated solver code was invalid Python: {syntax_error}",
                "model": "scientific_solver",
            }

        # Step 2: Execute Code
        exec_res = self._execute_code(code)
        
        if not exec_res["success"]:
            logger.error(f"Scientific execution failed: {exec_res['error']}")
            return {
                "success": False,
                "error": exec_res["error"],
                "model": "scientific_solver",
            }
        
        # Step 3: Split Synthesis (Parallel Brief/Detailed generation)
        brief_system = (
            "You are an expert scientific tutor. Create the 'BRIEF' part of a response.\n"
            "CRITICAL: DO NOT THINK. DO NOT USE <think> TAGS. START DIRECTLY WITH '{'.\n"
            "1. BRIEF VIEW (Quick Steps): COMPLETE mathematical solution. Address ALL parts of the question. DO NOT stop at intermediate steps. Reach the ULTIMATE final answer.\n"
            "2. FINAL_RESULT: Use the 'FINAL_RESULT' from the execution results to provide the ultimate answer.\n"
            "3. MATH FORMATTING: Use KaTeX ($...$ or $$...$$).\n\n"
            "JSON STRUCTURE:\n"
            "{\n"
            "  \"subject\": \"...\",\n"
            "  \"type\": \"...\",\n"
            "  \"brief\": {\n"
            "    \"question_nature\": \"numerical | theoretical\",\n"
            "    \"steps\": [{\"text\": \"Math/Result only step\"}, ...],\n"
            "    \"final_answer\": \"(a) ... (b) ...\"\n"
            "  }\n"
            "}"
        )
        
        detailed_system = (
            "You are an expert scientific tutor. Create the 'DETAILED' part of a response.\n"
            "CRITICAL: DO NOT THINK. DO NOT USE <think> TAGS. START DIRECTLY WITH '{'.\n"
            "1. DETAILED VIEW: COMPLETE step-by-step prose explanation + math derivations for EVERY part of the question. DO NOT skip the final calculation.\n"
            "2. TOTAL COMPLETION: Explicitly solve for $f(x)$, $g(x)$, and the final requested value (e.g., $|p+q|$).\n"
            "3. NO INTERNAL REASONING: Do not include <think> tags.\n"
            "4. MATH FORMATTING: Use KaTeX.\n\n"
            "JSON STRUCTURE:\n"
            "{\n"
            "  \"detailed\": {\n"
            "    \"explanation\": \"Full prose with derivations\",\n"
            "    \"final_answer\": \"Summary\",\n"
            "    \"verification\": \"...\",\n"
            "    \"key_concept\": \"...\"\n"
            "  },\n"
            "  \"key_concepts\": [...]\n"
            "}"
        )

        res_summary = json.dumps({
            "stdout": exec_res["stdout"][:1000] if exec_res["stdout"] else "",
            "variables": {k: str(v)[:500] for k, v in list(exec_res["results"].items())[:20]},
            "error": str(exec_res["error"])[:500] if exec_res["error"] else None
        }, default=str)
        
        common_user = (
            f"Question: {question}\n\n"
            f"Executed Code:\n{code}\n\n"
            f"Results:\n{res_summary}\n\n"
            f"Provide the requested JSON section."
        )

        tasks = [
            {"system_prompt": brief_system, "user_prompt": common_user, "max_tokens": 1536},
            {"system_prompt": detailed_system, "user_prompt": common_user, "max_tokens": 3072}
        ]
        
        results = self.llm.parallel_complete_many(
            tasks=tasks,
            model="llama-3.3-70b-versatile",
            json_mode=True
        )
        
        brief_resp = results[0]["content"]
        detailed_resp = results[1]["content"]
        
        # Merge results
        content = {
            "subject": brief_resp.get("subject", "Science"),
            "type": brief_resp.get("type", "Numerical"),
            "brief": brief_resp.get("brief", {}),
            "detailed": detailed_resp.get("detailed", {}),
            "key_concepts": detailed_resp.get("key_concepts", [])
        }
        
        # Step 4: Post-process to replace placeholders with actual base64 in all fields
        def replace_placeholders(text: Any) -> Any:
            if isinstance(text, str):
                for i, g_uri in enumerate(exec_res["graphs"]):
                    text = text.replace(f"IMAGE_PLACEHOLDER_{i}", g_uri)
                return text
            if isinstance(text, list):
                return [replace_placeholders(i) for i in text]
            if isinstance(text, dict):
                return {k: replace_placeholders(v) for k, v in text.items()}
            return text

        content = replace_placeholders(content)
        return content

scientific_solver = ScientificSolver()
