"""Synthetic enterprise corpus + a 100-question gold set.

195 documents (100 internal docs, 50 meeting notes, 25 contracts/SOWs,
20 sales/customer records) carrying specific, checkable facts. The
generator deliberately plants:

- exact facts (single doc, single value)
- the same fact paraphrased across doc types (near-repeat recall)
- multi-document chains (a value only obtainable by joining two docs)
- contradiction traps (a stale/low-authority doc disagreeing with an
  authoritative, more recent one)

Every fact is generated WITH its gold answer, so scoring is exact and
needs no human labelling.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

random.seed(7)

PROJECTS = ["Orion", "Helios", "Vega", "Atlas", "Nova", "Lyra", "Draco", "Cygnus",
            "Phoenix", "Titan", "Rhea", "Pavo", "Corvus", "Mensa", "Tucana"]
DEPARTMENTS = ["Platform", "Growth", "Security", "Data", "Infrastructure", "Mobile"]
VENDORS = ["Acme", "Globex", "Initech", "Umbrella", "Hooli", "Stark", "Wayne", "Soylent"]
CUSTOMERS = ["Northwind", "Contoso", "Fabrikam", "Tailspin", "Adventure", "Wingtip",
             "Litware", "Proseware", "Fourth Coffee", "Graphic Design Inc"]
OWNERS = ["Dana Reyes", "Sam Okafor", "Priya Nair", "Lars Eriksson", "Mei Tan",
          "Jamal Brooks", "Elena Costa", "Tom Whaley", "Aisha Khan", "Victor Ruiz"]


@dataclass
class Doc:
    doc_id: str
    kind: str
    title: str
    text: str
    date: str
    authority: float
    confidence: float


@dataclass
class QA:
    qid: str
    qtype: str           # exact | nearrepeat | synthesis | contradiction
    question: str
    answer: str          # gold substring that must appear
    source_docs: List[str] = field(default_factory=list)


class Corpus:
    def __init__(self):
        self.docs: List[Doc] = []
        self.qa: List[QA] = []
        self._n = 0

    def _id(self, kind):
        self._n += 1
        return f"{kind}-{self._n:03d}"

    def add(self, kind, title, text, date, authority, confidence) -> str:
        d = Doc(self._id(kind), kind, title, text, date, authority, confidence)
        self.docs.append(d)
        return d.doc_id


def build_corpus() -> Corpus:
    c = Corpus()
    proj_budget: Dict[str, int] = {}
    proj_owner: Dict[str, str] = {}
    proj_dept: Dict[str, str] = {}
    dept_vendor: Dict[str, str] = {}
    vendor_cost: Dict[str, int] = {}

    # ---- internal docs: project facts (100) ------------------------------
    for i, proj in enumerate(PROJECTS):
        budget = (i + 2) * 250_000
        owner = OWNERS[i % len(OWNERS)]
        dept = DEPARTMENTS[i % len(DEPARTMENTS)]
        proj_budget[proj] = budget
        proj_owner[proj] = owner
        proj_dept[proj] = dept
        c.add("doc", f"Project {proj} Charter",
              f"Project {proj} is owned by {owner}. Project {proj} belongs to the {dept} department. "
              f"The approved budget for Project {proj} is ${budget:,}.",
              "2026-01-%02d" % (i + 1), 0.85, 0.9)
        c.add("doc", f"Project {proj} Status",
              f"Project {proj} is currently on track. The {dept} department reviews Project {proj} weekly. "
              f"Project {proj} lead is {owner}.",
              "2026-02-%02d" % (i + 1), 0.7, 0.8)

    for dept in DEPARTMENTS:
        vendor = random.choice(VENDORS)
        dept_vendor[dept] = vendor
        c.add("doc", f"{dept} Vendor Plan",
              f"The {dept} department relies on vendor {vendor}. "
              f"All {dept} infrastructure runs through {vendor}.",
              "2026-01-15", 0.8, 0.85)

    for vendor in VENDORS:
        cost = random.randint(20, 90) * 1000
        vendor_cost[vendor] = cost
        c.add("doc", f"{vendor} Pricing Sheet",
              f"Vendor {vendor} charges ${cost:,} per year for the standard managed services tier.",
              "2026-01-20", 0.75, 0.8)

    # pad internal docs to ~100 with policy/process docs (noise + some facts)
    policies = [
        ("Security Policy", "All production access requires two-factor authentication and quarterly review."),
        ("Data Retention", "Customer data is retained for 24 months, then anonymized."),
        ("Incident Response", "Sev-1 incidents must be acknowledged within 15 minutes."),
        ("Release Process", "Releases ship on Tuesdays after passing the full regression suite."),
        ("Expense Policy", "Travel over $2,000 requires director approval."),
        ("Onboarding", "New hires complete security training in their first week."),
        ("Code Review", "Every change needs one approving review before merge."),
        ("On-call", "On-call rotations last one week and page via the duty phone."),
    ]
    pi = 0
    while len([d for d in c.docs if d.kind == "doc"]) < 100:
        title, body = policies[pi % len(policies)]
        c.add("doc", f"{title} v{pi//len(policies)+1}", body, "2026-01-10", 0.6, 0.7)
        pi += 1

    # ---- meeting notes (50): paraphrase facts (near-repeat) --------------
    for i in range(50):
        proj = PROJECTS[i % len(PROJECTS)]
        owner = proj_owner[proj]
        c.add("note", f"Standup {i+1} — {proj}",
              f"In today's sync, {owner} gave an update on the {proj} initiative. "
              f"The team confirmed {owner} continues to drive {proj}.",
              "2026-03-%02d" % ((i % 27) + 1), 0.5, 0.65)

    # ---- contracts / SOWs (25): vendor + cost + customer -----------------
    for i in range(25):
        vendor = VENDORS[i % len(VENDORS)]
        cust = CUSTOMERS[i % len(CUSTOMERS)]
        cost = vendor_cost[vendor]
        c.add("contract", f"SOW {i+1}: {cust} / {vendor}",
              f"This statement of work engages vendor {vendor} for customer {cust}. "
              f"The annual contract value with {vendor} is ${cost:,}. "
              f"The {cust} account is serviced under this SOW.",
              "2026-02-%02d" % ((i % 27) + 1), 0.85, 0.88)

    # ---- sales / customer records (20) -----------------------------------
    # One consistent tier per customer (each customer gets two records).
    cust_tier: Dict[str, str] = {cust: ["Enterprise", "Growth", "Starter"][j % 3]
                                 for j, cust in enumerate(CUSTOMERS)}
    for i in range(20):
        cust = CUSTOMERS[i % len(CUSTOMERS)]
        tier = cust_tier[cust]
        c.add("record", f"Account {cust} #{i+1}",
              f"Customer {cust} is on the {tier} plan. "
              f"The {cust} account renews annually.",
              "2026-03-%02d" % ((i % 27) + 1), 0.8, 0.82)

    # ---- contradiction traps: authoritative revision vs stale doc --------
    trap_specs = []
    for i in range(10):
        proj = PROJECTS[i]
        old = proj_budget[proj]
        new = old + 300_000
        # stale, low authority
        c.add("doc", f"Project {proj} Old Memo",
              f"An older memo lists the Project {proj} budget as ${old:,}.",
              "2026-01-01", 0.3, 0.5)
        # authoritative revision (more recent, higher authority)
        c.add("doc", f"Project {proj} Budget Revision",
              f"The Project {proj} budget was revised to ${new:,} by the finance committee.",
              "2026-04-%02d" % (i + 1), 0.95, 0.95)
        proj_budget[proj] = new  # the current truth
        trap_specs.append((proj, old, new))

    # ===================== QUESTIONS =====================================
    qn = 0
    def q(qtype, question, answer, srcs):
        nonlocal qn
        qn += 1
        c.qa.append(QA(f"q-{qn:03d}", qtype, question, answer, srcs))

    # 40 exact recall
    for proj in PROJECTS[:10]:
        q("exact", f"Who owns Project {proj}?", proj_owner[proj], [])
    for proj in PROJECTS[:10]:
        q("exact", f"Which department does Project {proj} belong to?", proj_dept[proj], [])
    for vendor in VENDORS:
        q("exact", f"How much does vendor {vendor} charge per year?",
          f"${vendor_cost[vendor]:,}", [])
    for cust in CUSTOMERS[:10]:
        q("exact", f"What plan is customer {cust} on?", cust_tier[cust], [])
    for dept in DEPARTMENTS[:2]:
        q("exact", f"Which vendor does the {dept} department use?", dept_vendor[dept], [])

    # 30 near-repeat / paraphrased (answer in notes, asked differently)
    for i in range(30):
        proj = PROJECTS[i % len(PROJECTS)]
        phrasings = [
            f"Who is leading the {proj} initiative these days?",
            f"Remind me who's driving {proj}?",
            f"Who's the point person on the {proj} effort?",
        ]
        q("nearrepeat", phrasings[i % 3], proj_owner[proj], [])

    # 20 multi-document synthesis (join two docs)
    for i in range(10):
        proj = PROJECTS[i]
        dept = proj_dept[proj]
        vendor = dept_vendor[dept]
        # Project -> dept (charter) -> vendor (vendor plan)
        q("synthesis", f"Which vendor does Project {proj} ultimately rely on?", vendor, [])
    for i in range(10):
        proj = PROJECTS[i]
        dept = proj_dept[proj]
        vendor = dept_vendor[dept]
        cost = vendor_cost[vendor]
        # Project -> dept -> vendor -> cost
        q("synthesis", f"What is the annual vendor cost behind Project {proj}?",
          f"${cost:,}", [])

    # 10 contradiction traps (must return the revised, authoritative value)
    for proj, old, new in trap_specs:
        q("contradiction", f"What is the current Project {proj} budget?", f"${new:,}", [])

    return c


if __name__ == "__main__":
    c = build_corpus()
    from collections import Counter
    print("docs:", Counter(d.kind for d in c.docs), "total", len(c.docs))
    print("questions:", Counter(x.qtype for x in c.qa), "total", len(c.qa))
