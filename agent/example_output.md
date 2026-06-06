# Multi-Agent Triage — Example Run (for the paper / screenshots)

Command:

```
python agent/crew_ticket_triage.py
```

## Input ticket

```
Subject: Cannot access the internet in the whole office

Since about 9am none of the computers on the 3rd floor can reach any website
and our VPN keeps disconnecting. We have an important client demo at noon.
Please help urgently.
```

## Agent 1 — Ticket Classification Specialist (calls the trained DistilBERT tool)

> Tool `ticket_classifier` →
> **Predicted department: Technical Support (33%)**, IT Support (29%),
> Customer Service (15%).

## Agent 2 — Support Triage Router

> **Route to Technical Support, Medium priority** — Technical Support is the most
> likely department and is best equipped for network/VPN problems.

## Agent 3 — Customer Response Writer (final output)

> Dear [Customer's Name],
>
> Thank you for reaching out to us about the issue with your internet connection on the
> 3rd floor. We have received your ticket and our Technical Support team will be handling
> this case, as they are best equipped to assist with network connectivity and VPN issues.
> You can expect a response from them within the next 2–4 hours during regular business
> hours. Please note that our team is working to resolve the issue before your important
> client demo at noon.
>
> Best regards,
> Technical Support Team

---

*Model:* this run uses the final classifier (7 epochs, full data) at
`models/distilbert-ticket-best`. The two top departments (Technical Support and IT
Support) are exactly the IT-related queues, which is correct for a network/VPN outage.
*To screenshot:* run the command above and capture the terminal.
