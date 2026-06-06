"""CrewAI triage crew: three agents handle one support ticket.

  1. Classifier agent - calls our DistilBERT model (classifier_tool.py) as a tool
  2. Router agent     - picks the team and priority
  3. Responder agent  - writes a short reply to the customer

The agents run on a local LLM via Ollama (no API key).

Needs: Ollama running, the ticket-triage-llm model (see agent/Modelfile), and a
trained model in models/distilbert-ticket-best.

  python agent/crew_ticket_triage.py
  python agent/crew_ticket_triage.py "My VPN won't connect and I can't reach email."
"""

from __future__ import annotations

import os
import sys

# make src/ importable so we can load the classifier
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import bigstack  # noqa: E402

# ticket-triage-llm = llama3.2:3b with a 4k context (fits the 8 GB GPU).
# build it with: ollama create ticket-triage-llm -f agent/Modelfile
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "ollama/ticket-triage-llm")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

EXAMPLE_TICKET = (
    "Subject: Cannot access the internet in the whole office\n\n"
    "Since about 9am none of the computers on the 3rd floor can reach any website "
    "and our VPN keeps disconnecting. We have an important client demo at noon. "
    "Please help urgently."
)


def main() -> None:
    # UTF-8 console so CrewAI's emoji logs don't crash on Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from typing import Type

    from crewai import Agent, Crew, LLM, Process, Task
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    from classifier_tool import classify_ticket

    # imports done - drop back to the normal stack so CrewAI's threads don't OOM
    import threading
    threading.stack_size(0)

    ticket = " ".join(sys.argv[1:]).strip() or EXAMPLE_TICKET

    # ---- 1) the TOOL that wraps our trained classifier ----------------------
    class ClassifyInput(BaseModel):
        ticket_text: str = Field(..., description="The full text of the support ticket.")

    class TicketClassifierTool(BaseTool):
        name: str = "ticket_classifier"
        description: str = (
            "Classify a customer-support ticket into one of 10 departments "
            "(Technical Support, IT Support, Billing and Payments, Customer Service, "
            "Product Support, Returns and Exchanges, Sales and Pre-Sales, "
            "Service Outages and Maintenance, Human Resources, General Inquiry) "
            "using a fine-tuned DistilBERT model. Returns the predicted department "
            "and a confidence score. Always use this tool to decide the department."
        )
        args_schema: Type[BaseModel] = ClassifyInput

        def _run(self, ticket_text: str = "") -> str:
            # classify the real ticket - the small LLM sometimes passes a fragment
            r = classify_ticket(ticket)
            top3 = list(r["scores"].items())[:3]
            top3_str = ", ".join(f"{lbl} ({sc:.0%})" for lbl, sc in top3)
            return (
                f"Predicted department: {r['label']} "
                f"(confidence {r['confidence']:.0%}). Top options: {top3_str}."
            )

    classifier_tool = TicketClassifierTool()

    # warm up the model on the main thread (loading it inside a CrewAI worker crashes)
    warmup = classify_ticket(ticket)
    print(f"[classifier] predicted department: {warmup['label']} "
          f"({warmup['confidence']:.0%})\n")

    # the 4k context is set in the model itself (agent/Modelfile)
    llm = LLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)

    # ---- 3) the three agents -----------------------------------------------
    classifier_agent = Agent(
        role="Ticket Classification Specialist",
        goal="Determine the correct department for an incoming support ticket.",
        backstory=(
            "You are precise and rely on the machine-learning ticket_classifier tool "
            "rather than guessing. You always call the tool with the ticket text."
        ),
        tools=[classifier_tool],
        llm=llm,
        verbose=True,
    )

    router_agent = Agent(
        role="Support Triage Router",
        goal="Route the ticket to the right team and assign a priority.",
        backstory=(
            "You are an experienced support team lead. Network/connectivity/VPN issues "
            "affecting many users are high priority. Billing issues go to Finance. "
            "You map the predicted department to a concrete team and a priority "
            "(Low / Medium / High / Critical)."
        ),
        llm=llm,
        verbose=True,
    )

    responder_agent = Agent(
        role="Customer Response Writer",
        goal="Write a short, professional acknowledgement to the customer.",
        backstory=(
            "You write concise, empathetic 3-4 sentence replies that confirm the issue "
            "was received, state which team is handling it, and set expectations."
        ),
        llm=llm,
        verbose=True,
    )

    # ---- 4) the tasks (run in sequence, each builds on the previous) --------
    classify_task = Task(
        description=(
            "A new support ticket has arrived:\n\n---\n{ticket}\n---\n\n"
            "Use the ticket_classifier tool to determine the department. "
            "Report the predicted department and confidence."
        ),
        expected_output="The predicted department name and its confidence score.",
        agent=classifier_agent,
    )

    route_task = Task(
        description=(
            "Based on the predicted department, decide which team should handle the "
            "ticket and assign a priority (Low/Medium/High/Critical). Give one short "
            "sentence of justification."
        ),
        expected_output="Team to route to + priority level + one-sentence reason.",
        agent=router_agent,
        context=[classify_task],
    )

    respond_task = Task(
        description=(
            "Write a short acknowledgement reply (3-4 sentences) to the customer for "
            "this ticket:\n\n---\n{ticket}\n---\n\n"
            "Mention that the issue was received and which team is handling it."
        ),
        expected_output="A short, polite customer-facing reply.",
        agent=responder_agent,
        context=[classify_task, route_task],
    )

    crew = Crew(
        agents=[classifier_agent, router_agent, responder_agent],
        tasks=[classify_task, route_task, respond_task],
        process=Process.sequential,
        verbose=True,
    )

    print("=" * 70)
    print("INPUT TICKET:")
    print(ticket)
    print("=" * 70)

    result = crew.kickoff(inputs={"ticket": ticket})

    print("\n" + "=" * 70)
    print("FINAL OUTPUT (customer reply):")
    print("=" * 70)
    print(result)


if __name__ == "__main__":
    bigstack.run(main)
