"""
CSP and non-CSP prompt datasets for probe training and evaluation.

These prompts are used to:
1. Train the CSP detection probe (Experiment 1)
2. Train the CSP subtype classifier (Experiment 2)
3. Evaluate end-to-end performance (Experiment 5-6)
"""

from typing import Iterator

# =============================================================================
# CSP PROMPTS - Organized by constraint satisfaction problem type
# =============================================================================

CSP_PROMPTS: dict[str, list[str]] = {
    "scheduling": [
        "Schedule meetings: Alice needs 2 hours, Bob needs 1 hour, they can't overlap",
        "Plan my day: gym before lunch, dentist at 2pm fixed, dinner after 6pm",
        "Assign time slots: Task A (2hr), Task B (1hr), Task C (1.5hr), window 9am-1pm",
        "Meeting room booking: 3 meetings, 2 rooms, no conflicts allowed",
        "Schedule my day: gym (1hr), lunch meeting (1.5hr), dentist at 2pm (1hr). Dentist is fixed.",
        "Create a weekly schedule: 5 classes, each 1 hour, no teacher teaches twice same day",
        "Plan project timeline: design (3 days), build (5 days), test (2 days), build after design",
        "Shift scheduling: 3 nurses, 2 shifts per day, each nurse works max 1 shift",
        "Exam timetable: 4 exams, 2 time slots, students can't have overlapping exams",
        "Interview schedule: 5 candidates, 3 interviewers, each candidate meets all interviewers",
    ],
    "assignment": [
        "Assign 4 developers to 2 projects, balance experience levels",
        "Match 5 students to 3 advisors, respect preferences",
        "Allocate nurses to shifts: 10 nurses, 3 shifts, minimum 3 per shift",
        "Team formation: 8 people into pairs, avoid past conflicts",
        "Assign developers Alice, Bob, Carol to projects X and Y. Alice can't work with Bob.",
        "Room assignment: 6 students, 3 double rooms, roommate preferences",
        "Task assignment: 5 tasks to 3 workers, each worker has different skills",
        "Match mentors to mentees: 4 mentors, 8 mentees, max 2 per mentor",
        "Assign reviewers to papers: 10 papers, 5 reviewers, no conflicts of interest",
        "Crew scheduling: 6 pilots, 4 flights, rest time requirements",
    ],
    "routing": [
        "Visit NYC, LA, Chicago, Seattle and return home, minimize distance",
        "Delivery route: 5 stops, truck capacity 100, each stop has demand",
        "Sales trip: visit 4 clients, some must be morning, minimize travel",
        "Campus tour: hit all 6 buildings, start and end at parking lot",
        "Visit offices in NYC, Boston, DC, return to NYC. Minimize travel.",
        "School bus route: pick up 8 students, max 30 min ride for any student",
        "Warehouse picking: collect 10 items, minimize total walking distance",
        "Food delivery: 6 orders, 2 drivers, each order has time window",
        "Service technician route: 7 customers, different service times, 8-hour day",
        "Mail carrier route: 50 houses, must return to post office, minimize distance",
    ],
    "packing": [
        "Pack items [3, 5, 2, 7, 4] into bins of capacity 10",
        "Load trucks: packages [20, 35, 15, 40], max 50 per truck",
        "Knapsack: items with values and weights, capacity 15kg, maximize value",
        "Container loading: boxes of different sizes, minimize containers used",
        "Cut stock: need pieces [3m, 5m, 2m, 4m] from 10m rods, minimize waste",
        "Pallet loading: 12 boxes, each pallet holds 4 boxes max weight 100kg",
        "Memory allocation: programs [128MB, 256MB, 64MB], RAM blocks of 512MB",
        "Bin packing: items weighing [8, 4, 2, 6, 3, 5] into bins of capacity 12",
        "Cargo loading: ship compartments [100t, 80t, 120t], cargo [30t, 50t, 40t, 60t]",
        "Disk scheduling: files [2GB, 5GB, 1GB, 3GB] onto 10GB disks",
    ],
    "coloring": [
        "Schedule exams: 5 courses, some share students, minimize time slots",
        "Frequency assignment: 4 towers, adjacent can't share frequency",
        "Register allocation: 6 variables, some live simultaneously, 3 registers",
        "Color a map: 4 regions, neighbors can't have same color, min colors",
        "Course scheduling: 6 courses, room conflicts, 4 time slots available",
        "Wavelength assignment: 5 optical channels, minimize wavelengths used",
        "Task coloring: 7 tasks with conflicts, assign to minimum workers",
        "Sports league scheduling: 6 teams, each plays each other, no same-day repeats",
        "Radio frequency: 8 transmitters, interference constraints, 4 frequencies",
        "Meeting room colors: 5 meetings with overlaps, assign to minimum rooms",
    ],
}

# =============================================================================
# NON-CSP PROMPTS - Various task types that should NOT trigger CSP expert
# =============================================================================

NON_CSP_PROMPTS: dict[str, list[str]] = {
    "arithmetic": [
        "What is 127 * 89?",
        "Calculate 456 + 789",
        "How much is 1024 divided by 16?",
        "Find the square root of 144",
        "What's 15% of 200?",
        "Compute 2^10",
        "Add up 1+2+3+4+5+6+7+8+9+10",
        "What is 999 minus 111?",
        "Multiply 25 by 4",
        "Divide 100 by 7, give me the remainder",
    ],
    "factual": [
        "What is the capital of France?",
        "Who invented the telephone?",
        "When did World War 2 end?",
        "What is the chemical formula for water?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet in our solar system?",
        "When was the Declaration of Independence signed?",
        "What language is spoken in Brazil?",
        "Who painted the Mona Lisa?",
        "What is the speed of light?",
    ],
    "creative": [
        "Write a haiku about rain",
        "Tell me a joke about programmers",
        "Describe a sunset in three sentences",
        "Create a limerick about coffee",
        "Write a short poem about autumn",
        "Make up a story about a lost key",
        "Describe what happiness feels like",
        "Write a product description for invisible ink",
        "Create a metaphor for time passing",
        "Write a six-word story",
    ],
    "code": [
        "Write a Python function to reverse a string",
        "How do I sort a list in JavaScript?",
        "Fix this bug: for i in range(10) print(i)",
        "What does the map function do in Python?",
        "Write a SQL query to find duplicates",
        "How do I read a file in C++?",
        "Explain recursion with an example",
        "What's the difference between == and === in JavaScript?",
        "Write a regex to match email addresses",
        "How do I create a class in Python?",
    ],
    "reasoning": [
        "Why is the sky blue?",
        "Explain how photosynthesis works",
        "What causes inflation?",
        "Why do we dream?",
        "How does a refrigerator work?",
        "What makes ice slippery?",
        "Why do leaves change color in fall?",
        "How do airplanes stay in the air?",
        "What causes earthquakes?",
        "Why is the ocean salty?",
    ],
    "conversational": [
        "Hello, how are you?",
        "Thanks for your help!",
        "What should I have for lunch?",
        "Tell me about yourself",
        "What's your favorite color?",
        "Can you help me?",
        "Good morning!",
        "I'm feeling bored today",
        "What do you think about AI?",
        "Nice to meet you",
    ],
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_all_csp_prompts() -> list[tuple[str, str]]:
    """
    Get all CSP prompts with their category labels.

    Returns:
        List of (prompt, category) tuples
    """
    result = []
    for category, prompts in CSP_PROMPTS.items():
        for prompt in prompts:
            result.append((prompt, category))
    return result


def get_all_non_csp_prompts() -> list[tuple[str, str]]:
    """
    Get all non-CSP prompts with their category labels.

    Returns:
        List of (prompt, category) tuples
    """
    result = []
    for category, prompts in NON_CSP_PROMPTS.items():
        for prompt in prompts:
            result.append((prompt, category))
    return result


def iter_prompts(include_csp: bool = True, include_non_csp: bool = True) -> Iterator[tuple[str, str, bool]]:
    """
    Iterate over prompts with labels.

    Yields:
        (prompt, category, is_csp) tuples
    """
    if include_csp:
        for prompt, category in get_all_csp_prompts():
            yield prompt, category, True

    if include_non_csp:
        for prompt, category in get_all_non_csp_prompts():
            yield prompt, category, False


def get_calibration_prompts() -> tuple[list[str], list[str]]:
    """
    Get prompts formatted for VirtualExpertPlugin calibration.

    Returns:
        (positive_prompts, negative_prompts) tuple
    """
    positive = [p for p, _ in get_all_csp_prompts()]
    negative = [p for p, _ in get_all_non_csp_prompts()]
    return positive, negative


# =============================================================================
# STATISTICS
# =============================================================================

if __name__ == "__main__":
    print("CSP Prompt Dataset Statistics")
    print("=" * 50)

    print("\nCSP Prompts:")
    total_csp = 0
    for category, prompts in CSP_PROMPTS.items():
        print(f"  {category}: {len(prompts)}")
        total_csp += len(prompts)
    print(f"  TOTAL: {total_csp}")

    print("\nNon-CSP Prompts:")
    total_non_csp = 0
    for category, prompts in NON_CSP_PROMPTS.items():
        print(f"  {category}: {len(prompts)}")
        total_non_csp += len(prompts)
    print(f"  TOTAL: {total_non_csp}")

    print(f"\nGrand Total: {total_csp + total_non_csp} prompts")
    print(f"Class balance: {total_csp}/{total_non_csp} ({total_csp/(total_csp+total_non_csp):.1%} CSP)")
