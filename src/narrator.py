import os


class EvacuationNarrator:
    """
    Single-loop LLM narrator.

    It generates:
    1. One sentence every 30 steps.
    2. A final 3-paragraph chronological story.
    """

    def __init__(self, model="llama3.2:3b"):
        self.model = model
        self.timeline = []

    def _call_ollama(self, prompt):
        """
        Calls local Ollama model.

        If Ollama is not running, a fallback text is returned so the project
        can still run during testing.
        """
        try:
            import ollama

            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a concise evacuation narrator. "
                            "Use only the given simulation metrics. "
                            "Do not invent numbers."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            return response["message"]["content"].strip()

        except Exception:
            return None

    def narrate_step(self, step, tool_summary):
        """
        Produces one short sentence during the simulation.
        """
        prompt = f"""
Generate exactly one short sentence describing the evacuation at step {step}.

Use only these metrics:
{tool_summary}

Mention congestion, exit load, or peak density if relevant.
Do not add unsupported details.
"""

        llm_text = self._call_ollama(prompt)

        if llm_text is None:
            exit_loads = tool_summary["exit_loads"]["exit_loads"]
            peak_density = tool_summary["peak_area"]["peak_density"]

            llm_text = (
                f"At step {step}, the evacuation continued with exit loads "
                f"{exit_loads} and a peak local density of {peak_density}."
            )

        self.timeline.append(
            {
                "step": step,
                "sentence": llm_text,
                "tool_summary": tool_summary,
            }
        )

        return llm_text

    def final_story(self, final_metrics):
        """
        Produces a final 3-paragraph chronological evacuation story.
        """
        timeline_text = "\n".join(
            [
                f"Step {item['step']}: {item['sentence']}"
                for item in self.timeline
            ]
        )

        prompt = f"""
Write a 3-paragraph chronological story of the evacuation.

Use only the following timeline and final metrics.

Timeline:
{timeline_text}

Final metrics:
{final_metrics}

Paragraph 1: beginning of the evacuation.
Paragraph 2: middle phase and congestion behavior.
Paragraph 3: final outcome and performance.

Do not invent numbers.
Keep it concise and academic.
"""

        llm_text = self._call_ollama(prompt)

        if llm_text is None:
            llm_text = f"""
At the beginning of the evacuation, pedestrians selected exits and started moving through the continuous 2D room. The simulation tracked exit loads, local density, and evacuation progress over time.

During the middle phase, congestion was monitored using the tool outputs collected every 30 steps. The narrator focused on real metrics such as exit load and peak crowd density instead of unsupported assumptions.

By the end of the episode, {final_metrics['evacuated_count']} out of {final_metrics['n_pedestrians']} pedestrians evacuated successfully. The completion rate was {final_metrics['completion_rate']:.2f}, and the average evacuation time was {final_metrics['avg_evacuation_time']:.2f} steps.
""".strip()

        return llm_text

    def save_story(self, story, path="outputs/evacuation_story.md"):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            file.write("# Evacuation Story\n\n")
            file.write(story)

        return path