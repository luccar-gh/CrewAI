import os
import time
from crewai import Agent, Task, Crew, Process, LLM

llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0.7
)

# --- GAME CONTEXT ---
GAME_RULES = """
WORKSHOP GAME: Mapping friction in the Danish municipal aid assessment process for visually impaired citizens.
GAME STEPS: (1) Identify stakeholders & connections, (2) Walk through scenario with Task Trackers 
(INPUT/OUTPUT per step), (3) Place Lightning Bolts on friction points, (4) Brainstorm solutions with Method Cards.
SCENARIO: Visually impaired citizen applies for a screen reader after ophthalmologist referral.
CONTEXT: Legal deadline is 8 weeks but takes 6-12 months. Multiple stakeholders, every municipality is different.
"""

# --- 2. AGENTS (compact backstories to save tokens) ---

facilitator = Agent(
    role='Game Facilitator',
    goal='Guide participants through game phases, collect observations about game usability.',
    backstory=f'You run this workshop game. Guide participants, ensure they use Task Trackers and Lightning Bolts correctly. Note if instructions are unclear. {GAME_RULES}',
    verbose=True, allow_delegation=True, llm=llm
)

caseworker = Agent(
    role='Municipal Caseworker (Hilleroed)',
    goal='Share your workflow and identify friction points honestly.',
    backstory=f'Caseworker at Hilleroed Municipality. You receive applications (often to wrong dept), need medical docs from doctors (causes weeks of delay), handle high volume, rarely meet 8-week deadline, use different IT system than communication center. First time playing. {GAME_RULES}',
    verbose=True, allow_delegation=False, llm=llm
)

comm_specialist = Agent(
    role='Communication Center Specialist (Rudersdal)',
    goal='Show how you interact with municipalities and where communication breaks down.',
    backstory=f'Specialist at Rudersdal Communication Center. You do expert assessments, reports take 3-4 weeks due to workload, use different data system than municipalities, sometimes receive incomplete referrals. Formal and detail-oriented. First time playing. {GAME_RULES}',
    verbose=True, allow_delegation=False, llm=llm
)

citizen_rep = Agent(
    role='Visually Impaired Citizen',
    goal='Share your lived experience and identify where the system fails for users.',
    backstory=f'Visually impaired, applied for screen reader, took 9 months. Doctor didnt explain process, couldnt read application form, sent to wrong department first, returned without saying which is right, waited 3 months no update, couldnt reach caseworker by phone, eventually sent to communication center (2 more months). Frustrated but willing to help. {GAME_RULES}',
    verbose=True, allow_delegation=False, llm=llm
)

doctor = Agent(
    role='Ophthalmologist',
    goal='Explain your referral role and where medical-municipal communication breaks down.',
    backstory=f'Ophthalmologist who refers patients. You write referrals but unsure which municipal dept handles visual aid. Give patients general referral without process guidance. Municipalities ask for more docs weeks later, your response takes 1-2 weeks. No access to municipal systems. Brief and professional. {GAME_RULES}',
    verbose=True, allow_delegation=False, llm=llm
)

# --- 3. TASKS (Combined into 2 phases to stay within rate limits) ---

phase_1_mapping_and_walkthrough = Task(
    description="""COMBINED PHASE 1: STAKEHOLDER MAPPING + SCENARIO WALKTHROUGH + FRICTION POINTS

    Guide participants through the game. Do the following in ONE pass:

    1. Ask EACH stakeholder (caseworker, comm specialist, citizen, doctor) using ask_question_to_coworker:
       - Their role and who they interact with
       - Walk through their part of the scenario (citizen applies for screen reader)
       - For each step: what INPUT they need, what OUTPUT they produce, how long it takes
       - Where they experience FRICTION (delays, bottlenecks, miscommunication)
       - Place Lightning Bolt markers on friction points

    Keep questions concise. Collect answers from all 4 participants.

    Then compile:
    - Complete stakeholder map
    - Step-by-step workflow with Task Tracker data
    - ALL friction points ranked by severity
    - Patterns mentioned by multiple stakeholders""",
    expected_output='Stakeholder map, step-by-step workflow with input/output/duration, and ranked friction points with Lightning Bolt markers.',
    agent=facilitator
)

phase_2_solutions_and_evaluation = Task(
    description="""COMBINED PHASE 2: SOLUTIONS + GAME EVALUATION

    Based on Phase 1 findings, do TWO things:

    A) SOLUTIONS - Present these Method Cards and ask each stakeholder which would help most:
       Card 1: Eliminate Handoffs (combine steps)
       Card 2: Single Point of Contact (one coordinator)
       Card 3: Standardize Inputs (clear requirements at each step)
       Card 4: Parallel Processing (simultaneous steps)
       Card 5: Shared Information System (shared platform)
    
    Ask each stakeholder using ask_question_to_coworker which card(s) they pick and why.

    B) GAME EVALUATION - Also ask each stakeholder:
       - Were the game instructions clear?
       - Was Task Tracker easy to use?
       - Did Lightning Bolt system work?
       - Were Method Cards understandable for non-engineers?
       - What was missing or should be simpler?

    Compile: Top 3 solutions, game usability feedback, concrete improvements.""",
    expected_output='Top 3 solutions with stakeholder support, game usability evaluation, and concrete improvement suggestions.',
    agent=facilitator,
    context=[phase_1_mapping_and_walkthrough]
)

# --- 4. RUN ---

game = Crew(
    agents=[facilitator, caseworker, comm_specialist, citizen_rep, doctor],
    tasks=[phase_1_mapping_and_walkthrough, phase_2_solutions_and_evaluation],
    process=Process.sequential,
    verbose=True
)

print("=" * 70)
print("GAME SIMULATION: Stakeholder Friction Mapping Workshop")
print("=" * 70)
print("Scenario: Visually impaired citizen applies for screen reader")
print("Players: Facilitator, Caseworker, Comm Specialist, Citizen, Doctor")
print("=" * 70 + "\n")

result = game.kickoff()

print("\n" + "=" * 70)
print("FINAL SIMULATION REPORT")
print("=" * 70)
print(result)

# Save result to file
with open("simulation_result.txt", "w") as f:
    f.write(str(result))
print("\nResult saved to simulation_result.txt")
