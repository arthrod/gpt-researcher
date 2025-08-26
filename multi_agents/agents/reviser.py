from .utils.llms import call_model
<<<<<<< HEAD
from .utils.views import print_agent_output
=======
>>>>>>> 1027e1d0 (Fix linting issues)

sample_revision_notes = """
{
  "draft": {
    draft title: The revised draft that you are submitting for review
  },
  "revision_notes": Your message to the reviewer about the changes you made to the draft based on their feedback
}
"""


class ReviserAgent:
    def __init__(self, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}

    async def revise_draft(self, draft_state: dict):
        """
        Revise a draft using reviewer notes by calling the configured language model and returning the model's JSON response.
        
        Given a draft_state dict (expected keys described below), builds a system+user prompt that instructs the model to produce a revised draft and reviewer-style revision notes, then calls the LLM and returns its parsed JSON response.
        
        Parameters:
            draft_state (dict): Input state containing:
                - "review" (str): Reviewer's notes to guide the revision.
                - "task" (dict): Task configuration; must include a "model" key naming the model to call.
                - "draft" (str or dict): The original draft to be revised.
        
        Returns:
            dict: Parsed JSON response from the model following the expected sample_revision_notes format:
                - "draft" (object/string): The revised draft produced by the model.
                - "revision_notes" (str): Reviewer's-style notes describing the changes and rationale.
        """
        review = draft_state.get("review")
        task = draft_state.get("task")
        draft_report = draft_state.get("draft")
        prompt = [
            {
                "role": "system",
                "content": "You are an expert writer. Your goal is to revise drafts based on reviewer notes.",
            },
            {
                "role": "user",
                "content": f"""Draft:\n{draft_report}" + "Reviewer's notes:\n{review}\n\n
You have been tasked by your reviewer with revising the following draft, which was written by a non-expert.
If you decide to follow the reviewer's notes, please write a new draft and make sure to address all of the points they raised.
Please keep all other aspects of the draft the same.
You MUST return nothing but a JSON in the following format:
{sample_revision_notes}
""",
            },
        ]

        response = await call_model(
            prompt,
            model=task.get("model"),
            response_format="json",
        )
        return response

    async def run(self, draft_state: dict):
        """
        Run the reviser agent: request a revised draft from the model, optionally stream or print revision notes, and return the updated draft and notes.
        
        Given a draft_state (expected to include keys "draft", "review", and "task"), this coroutine calls self.revise_draft(draft_state) to obtain a revision. If task.get("verbose") is truthy, the method will either stream the revision notes over self.websocket using self.stream_output (if both are provided) or print them via print_agent_output. Returns a dict with keys:
        - "draft": the revised draft (from the model response)
        - "revision_notes": the reviewer's revision notes (from the model response)
        """
        print_agent_output("Rewriting draft based on feedback...", agent="REVISOR")
        revision = await self.revise_draft(draft_state)

        if draft_state.get("task").get("verbose"):
            if self.websocket and self.stream_output:
                await self.stream_output(
                    "logs",
                    "revision_notes",
                    f"Revision notes: {revision.get('revision_notes')}",
                    self.websocket,
                )
            else:
                print_agent_output(
                    f"Revision notes: {revision.get('revision_notes')}", agent="REVISOR"
                )

        return {
            "draft": revision.get("draft"),
            "revision_notes": revision.get("revision_notes"),
        }
