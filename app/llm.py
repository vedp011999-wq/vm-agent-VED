"""Model clients. Both speak the OpenAI tool-calling format, so the agent does not care which one runs.

MockLLM is a small rule-based planner: it lets the app, tests and CI run offline and deterministically.
Real reasoning quality is measured by running the evals with OpenAILLM.
"""
import json
import re
import uuid
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class LLMResponse:
    tool_calls: list[ToolCall] = field(default_factory=list)
    content: str | None = None


class LLMDown(Exception):
    pass


class OpenAILLM:
    def __init__(self, model: str, base_url: str | None, api_key: str | None):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key or "none", max_retries=0, timeout=60)
        self.model = model

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        try:
            r = self.client.chat.completions.create(model=self.model, messages=messages, tools=tools, temperature=0)
        except Exception as e:
            raise LLMDown(str(e)) from e
        msg = r.choices[0].message
        calls = [ToolCall(tc.id or uuid.uuid4().hex[:8], tc.function.name, tc.function.arguments or "{}")
                 for tc in msg.tool_calls or []]
        return LLMResponse(calls, msg.content)


class MockLLM:
    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        goal = messages[1]["content"].lower()
        done = self._results(messages)  # tool name -> list of results seen so far

        def call(name, **args):
            return ToolCall(f"call_{uuid.uuid4().hex[:8]}", name, json.dumps(args))

        def finish(text):
            return LLMResponse([call("finish", summary=text)])

        if messages[-1]["content"] and messages[-1]["content"].startswith("APPROVAL RESULTS"):
            return finish("Final report. " + messages[-1]["content"].split("\n", 1)[1])
        if any(w in goal for w in ("delete", "terminate", "shut down", "destroy")):
            return finish("I can't do that: I can only propose one-step resizes for human approval. "
                          "No actions were proposed.")
        if not any(w in goal for w in ("vm", "resize", "rightsiz", "cost", "utiliz", "server")):
            return finish("Out of scope: I only analyse VM utilization and propose resizes.")

        if "list_vms" not in done:
            env = next((e for e in ("prod", "staging", "dev") if re.search(rf"\b{e}", goal)), None)
            return LLMResponse([call("list_vms", **({"environment": env} if env else {}))])

        listed = done["list_vms"][0].get("vms", [])
        candidates = [v["id"] for v in listed if v["signal"] in ("underutilized", "overutilized")]
        have_metrics = {m["id"]: m for m in done.get("get_vm_metrics", []) if "id" in m}
        missing = [c for c in candidates if c not in have_metrics]
        if missing and "get_vm_metrics" not in done:  # evidence before action
            return LLMResponse([call("get_vm_metrics", vm_id=v) for v in missing])

        sizes = ["small", "medium", "large", "xlarge"]
        if "propose_resize" not in done:
            proposals = []
            for m in have_metrics.values():
                i = sizes.index(m["size"]) + (-1 if m["signal"] == "underutilized" else 1)
                if 0 <= i < len(sizes):
                    reason = f"peak CPU {m['cpu_peak']}%, peak memory {m['mem_peak']}%"
                    proposals.append(call("propose_resize", vm_id=m["id"], target_size=sizes[i], reason=reason))
            if proposals:
                return LLMResponse(proposals)

        proposed = [f"{p['vm_id']} {p['from_size']}->{p['to_size']}" for p in done.get("propose_resize", [])
                    if "vm_id" in p]
        notes = [f"{v['id']} skipped: stale data" for v in listed if v["is_stale"]]
        notes += [f"{c} skipped: could not read metrics" for c in missing]
        text = (f"Proposed for approval: {', '.join(proposed)}." if proposed else "No resizes needed.")
        return finish(" ".join([text] + notes))

    @staticmethod
    def _results(messages: list[dict]) -> dict[str, list[dict]]:
        names = {tc["id"]: tc["function"]["name"]
                 for m in messages if m["role"] == "assistant" for tc in m.get("tool_calls") or []}
        out: dict[str, list[dict]] = {}
        for m in messages:
            if m["role"] == "tool":
                r = json.loads(m["content"])
                out.setdefault(names[m["tool_call_id"]], []).append(r.get("data") or {"error": r.get("error")})
        return out


def build_llm(settings):
    if settings.llm_provider == "openai":
        return OpenAILLM(settings.llm_model, settings.llm_base_url, settings.llm_api_key)
    return MockLLM()
