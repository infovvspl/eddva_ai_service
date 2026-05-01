import os
import io
import base64
import json
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
from typing import Any, Dict, Optional, List
from ai_services.core.llm_client import get_llm

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
        }

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
        """Executes Python code in a restricted environment and captures output."""
        stdout_capture = io.StringIO()
        results = {}
        
        # Prepend standard imports to ensure they are available even if LLM forgets
        full_code = (
            "import numpy as np\n"
            "import sympy as sp\n"
            "import matplotlib.pyplot as plt\n"
            "from scipy import integrate, optimize, stats\n"
            "from rdkit import Chem\n"
            "from rdkit.Chem import Descriptors, AllChem\n"
            "import pint\n"
            "import pubchempy as pcp\n"
            "ureg = pint.UnitRegistry()\n"
            "\n"
        ) + code
        
        # Prepare environment
        exec_globals = {
            "__builtins__": __builtins__,
            "print": lambda *args: stdout_capture.write(" ".join(map(str, args)) + "\n"),
        }
        
        try:
            # Clear any previous plots
            plt.clf()
            
            # Execute
            exec(full_code, exec_globals)
            
            # Capture variables defined in code (excluding builtins/modules)
            excluded_keys = {"__builtins__", "print", "np", "sp", "plt", "integrate", "optimize", "stats", "Chem", "Descriptors", "AllChem", "pint", "pcp", "ureg"}
            for k, v in exec_globals.items():
                if k not in excluded_keys and not k.startswith("_"):
                    # Only keep simple serializable types or convert to str
                    try:
                        # Attempt to check if it's a basic type
                        if isinstance(v, (int, float, str, list, dict, bool, type(None))):
                            json.dumps(v) # Test serialization
                            results[k] = v
                        else:
                            results[k] = str(v)
                    except:
                        results[k] = str(v)

            # Capture Matplotlib figures
            figs_base64 = []
            for i in plt.get_fignums():
                fig = plt.figure(i)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                figs_base64.append(f"data:image/png;base64,{img_str}")
                buf.close()

            return {
                "success": True,
                "stdout": stdout_capture.getvalue(),
                "results": results,
                "graphs": figs_base64,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": stdout_capture.getvalue(),
                "results": results,
                "graphs": [],
                "error": f"{str(e)}\n{traceback.format_exc()}"
            }

    async def solve(self, question: str, mode: str = "detailed") -> Dict[str, Any]:
        """Main entry point: Generate code -> Execute -> Synthesize answer."""
        
        # Step 1: Generate Solution Code
        system_prompt = (
            "You are a Scientific Code Generator for a JEE/NEET/CBSE tutor app. "
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
            "1. Focus on accuracy. Use SymPy for derivations if possible.\n"
            "2. If a plot is helpful, use plt.plot() etc. Do NOT use plt.show().\n"
            "3. Store the final answer in a variable named 'final_answer'.\n"
            "4. Respond ONLY with valid Python code. No markdown fences. No explanation."
        )
        
        user_prompt = f"Question: {question}\n\nWrite the Python code to solve this."
        
        llm_resp = self.llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="llama-3.3-70b-versatile",
            json_mode=False
        )
        
        code = llm_resp["content"].strip()
        # Clean up code if LLM included fences
        if code.startswith("```python"):
            code = code.split("```python")[1].split("```")[0].strip()
        elif code.startswith("```"):
            code = code.split("```")[1].split("```")[0].strip()

        # Step 2: Execute Code
        exec_res = self._execute_code(code)
        
        if not exec_res["success"]:
            logger.error(f"Scientific execution failed: {exec_res['error']}")
            # Fallback or retry? For now, we'll return the error to the synthesizer
        
        # Step 3: Synthesize Final Explanation
        synth_system = (
            "You are an expert scientific tutor. You will be given a question, "
            "the Python code used to solve it, and the execution results (output, variables).\n"
            "Your task is to provide a highly detailed, accurate response in a structured JSON format.\n\n"
            "CONTENT RULES:\n"
            "1. ADDRESS ALL SUB-PARTS: Use labels like (a), (b), (i), (ii) explicitly in the explanation AND the final_answer.\n"
            "2. MANDATORY LABELING: The 'final_answer' MUST be a labeled list of results, e.g., '(a)(i) Rate increases 4-fold; (a)(ii) Order is 2; (b) Time is 99.6 min'. NEVER omit the labels (a), (b), etc.\n"
            "3. BRIEF VIEW (Quick Steps): Use math/results ONLY. Minimal to no prose. (e.g., '$x = 5$').\n"
            "4. DETAILED VIEW: Use full prose explanations along with math derivations.\n"
            "5. NO TRAILING BACKSLASHES: Never end a line with a literal backslash '\\'.\n"
            "6. MATH FORMATTING: Use KaTeX ($...$ or $$...$$). NEVER use plain text for math symbols.\n\n"
            "JSON STRUCTURE:\n"
            "{\n"
            "  \"subject\": \"...\",\n"
            "  \"type\": \"...\",\n"
            "  \"brief\": {\n"
            "    \"question_nature\": \"numerical | theoretical\",\n"
            "    \"steps\": [{\"text\": \"Math/Result only step (no prose)\"}, ...],\n"
            "    \"final_answer\": \"(a) ... (b) ... (summary of all parts with labels)\"\n"
            "  },\n"
            "  \"detailed\": {\n"
            "    \"explanation\": \"Full step-by-step prose explanation with part labels (a), (b), etc.\",\n"
            "    \"final_answer\": \"(a) ... (b) ... (formal summary)\",\n"
            "    \"verification\": \"...\",\n"
            "    \"key_concept\": \"...\"\n"
            "  },\n"
            "  \"key_concepts\": [...]\n"
            "}"
        )
        
        res_summary = json.dumps({
            "stdout": exec_res["stdout"],
            "variables": exec_res["results"],
            "error": exec_res["error"]
        }, default=str)
        
        synth_user = (
            f"Question: {question}\n\n"
            f"Executed Code:\n{code}\n\n"
            f"Results:\n{res_summary}\n\n"
            f"Graphs generated: {len(exec_res['graphs'])} graphs.\n"
            f"Provide the complete structured response addressing all parts of the question."
        )
        
        final_resp = self.llm.complete(
            system_prompt=synth_system,
            user_prompt=synth_user,
            model="llama-3.3-70b-versatile",
            json_mode=True,
            json_mode_suffix="\n\nRespond with a JSON object following the structure above."
        )
        
        content = final_resp["content"]
        
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
