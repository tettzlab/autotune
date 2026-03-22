"""
Prompt chain -- definition and execution.

This is the ONLY file the agent modifies. Everything is fair game:
  - Prompts, number of steps, step order
  - Model selection per step (gpt-5-mini, gpt-5-nano, sonnet-4.6, haiku-4.5)
  - Reasoning effort per step ("low", "medium", "high")
  - Temperature, max_tokens
  - Dataflow: sequential, branching, parallel, critique loops, whatever
  - Any Python logic for orchestrating the chain

The ONE rule: all LLM calls MUST use `call_llm()` from llm.py.
This ensures token usage and cost are tracked accurately.

    from llm import call_llm
    response = await call_llm(messages=[...], model="gpt-5-mini")
"""

from llm import call_llm


async def run_chain(inputs: dict[str, str]) -> str:
    """Run the prompt chain. Return the final output text.

    Args:
        inputs: A dict of input fields as defined in task.yaml.
                For the novel plot outline task:
                  - inputs["premise"]: The story's core concept and central conflict.
                  - inputs["genre"]: Genre and tone description.
                  - inputs["characters"]: Markdown describing main characters.

    Returns:
        The generated plot outline as a string.
    """
    premise = inputs["premise"]
    genre = inputs["genre"]
    characters = inputs["characters"]

    # Step 1: Analyze + plan structure (single call, low effort)
    analysis = await call_llm(
        messages=[
            {"role": "system", "content": "You are a veteran fiction editor and story analyst."},
            {"role": "user", "content": (
                "Analyze these story elements and design the structural backbone.\n\n"
                f"## Premise\n{premise}\n\n"
                f"## Genre\n{genre}\n\n"
                f"## Characters\n{characters}\n\n"
                "Provide:\n"
                "1. Central conflict and stakes\n"
                "2. Each character's arc (want vs need, transformation)\n"
                "3. Thematic questions raised\n"
                "4. Three-act breakdown with act breaks\n"
                "5. Inciting incident, midpoint reversal, climax, resolution\n"
                "6. At least 2 subplot threads and where they intersect main plot\n"
                "7. Stakes escalation plan (3+ escalation points)\n"
                "8. Antagonist motivation and parallel journey\n"
                "9. Genre conventions to honor + one fresh subversion"
            )},
        ],
        model="gpt-5-nano",
        reasoning_effort="low",
        max_tokens=2048,
    )

    # Step 2: Generate full plot outline (high effort for quality)
    outline = await call_llm(
        messages=[
            {"role": "system", "content": (
                "You are an expert fiction editor who writes detailed, compelling "
                "plot outlines. Your outlines are known for vivid scene specificity, "
                "emotionally resonant character arcs, and structural precision."
            )},
            {"role": "user", "content": (
                f"## Story Analysis & Structure\n{analysis}\n\n"
                f"## Premise\n{premise}\n\n"
                f"## Genre\n{genre}\n\n"
                f"## Characters\n{characters}\n\n"
                "Create a complete, detailed plot outline with ALL of the following:\n\n"
                "1. **Story Overview** - One-paragraph hook\n"
                "2. **Theme** - Central thematic question explored through plot\n"
                "3. **Act I: Setup** - World, characters, inciting incident, act break\n"
                "4. **Act II: Confrontation** - Rising action, midpoint reversal, "
                "complications, subplots, stakes escalation\n"
                "5. **Act III: Resolution** - Climax, falling action, resolution, final image\n"
                "6. **Character Arcs** - How each major character transforms\n"
                "7. **Key Scenes** - At least 12 specific pivotal scenes with concrete "
                "actions, dialogue hooks, sensory details, emotional beats\n"
                "8. **Subplots** - At least 2 subplot threads woven into main plot\n"
                "9. **Antagonist Thread** - Antagonist's parallel journey and motivation\n"
                "10. **Emotional Map** - Reader's emotional journey: tension, relief, "
                "dread, hope, catharsis\n\n"
                "Write in vivid, specific language. Every scene should feel real. "
                "Use markdown formatting."
            )},
        ],
        model="gpt-5-mini",
        reasoning_effort="high",
        max_tokens=8192,
    )

    # Step 3: Self-critique and revise in one call (medium effort)
    revised = await call_llm(
        messages=[
            {"role": "system", "content": (
                "You are a demanding fiction editor. First critique the outline below "
                "against these criteria, then output the complete revised outline "
                "fixing all weaknesses. Output ONLY the revised outline."
            )},
            {"role": "user", "content": (
                f"## Plot Outline to Review\n{outline}\n\n"
                "Criteria to check:\n"
                "1. Clear three-act structure with identifiable act breaks\n"
                "2. Each major character has distinct arc (want, goal, transformation)\n"
                "3. Central conflict drives plot from inciting incident to climax\n"
                "4. At least 10 specific scenes with concrete actions\n"
                "5. Stakes escalate with 3+ worsening moments\n"
                "6. Theme woven through events and choices, not just stated\n"
                "7. Antagonist has coherent, non-trivial motivation\n"
                "8. At least 2 subplots connected to main plot\n"
                "9. Genre conventions respected with a fresh element\n"
                "10. Specific emotional turning points, not just mechanics\n\n"
                "Internally critique each criterion, then output the complete "
                "improved plot outline with all 10 sections."
            )},
        ],
        model="gpt-5-mini",
        reasoning_effort="medium",
        max_tokens=8192,
    )

    return revised
