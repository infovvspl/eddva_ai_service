# TODO: `tests_metering` shares Redis db 0 with the running service

**Status:** open, pre-existing, not caused by the PPT V2 work.
**First recorded:** 2026-10-01, during PPT V2 Phase 1.

## Symptom

Five tests in `ai_services/tests_metering.py` fail whenever a dev server (or
another test run) has recently used the same Redis instance:

```
FAIL: MeteredDecoratorTests.test_proceeds_when_concurrency_gate_errors
FAIL: TenantGateTests.test_allows_under_hard_cap
FAIL: TenantGateTests.test_budget_is_per_tenant
FAIL: UsageAccountingTests.test_booked_once_not_twice
FAIL: UsageAccountingTests.test_log_usage_books_tokens_against_the_budget
```

They fail two ways: a `429` where `None` was expected (the tenant gate thinks
the budget is spent), and a token assertion that is wrong by a growing amount.

## Why it is the environment and not the code

The expected value is constant; the actual value climbs on every run:

| Run | Expected | Actual |
|---|---|---|
| 1 | 1500 | 11600 |
| 2 | 1500 | 15900 |
| 3 | 1500 | 17400 |
| 4 | 1500 | 18900 |

A counter that accumulates across separate test processes is shared state, not
a logic error. `.env` sets `REDIS_URL=redis://localhost:6379/0`, Django's test
runner only isolates the *database*, and the metering tests book real tokens
against the real Redis keys — which the dev server is also writing to.

## Fix

Point tests at their own Redis logical database and flush it per run, e.g. a
test-only override of `REDIS_URL` to `redis://localhost:6379/15` plus a
`flushdb` in `setUp`. The accumulating keys are the budget/booking counters
under the tenant id used by those tests.

**Do not** "fix" this by raising the expected numbers — the assertions are
correct and the isolation is what is broken.

## Scope note

Deliberately excluded from the PPT V2 feature work so an unrelated test-harness
change does not ride along in a feature branch. The V2 branch adds 70 tests and
leaves this count unchanged at exactly these 5 failures.
