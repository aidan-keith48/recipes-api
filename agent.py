"""PR review agent. Runs in GitHub Actions (see .github/workflows/ci.yml).

Reads GITHUB_TOKEN, REPOSITORY ("owner/name"), PR_NUMBER and GROQ_API_KEY from the environment.
"""
import asyncio
import os
from typing import Any

import dotenv
from github import Auth, Github
from llama_index.core.agent.workflow import (
    AgentOutput,
    AgentWorkflow,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.groq import Groq

dotenv.load_dotenv()

github_token = os.getenv("GITHUB_TOKEN")
git = Github(auth=Auth.Token(github_token)) if github_token else Github()
repository = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")
_repo = git.get_repo(repository) if repository else None

llm = Groq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)


def get_pr_details(pr_number: int) -> dict[str, Any]:
    """Useful for getting the details of a pull request given its number.
    Returns the author, title, body, diff_url, state, head_sha and the list of commit SHAs."""
    pull_request = _repo.get_pull(pr_number)

    commit_SHAs = []
    for c in pull_request.get_commits():
        commit_SHAs.append(c.sha)

    return {
        "user": pull_request.user.login,
        "title": pull_request.title,
        "body": pull_request.body,
        "diff_url": pull_request.diff_url,
        "state": pull_request.state,
        "head_sha": pull_request.head.sha,
        "commit_SHAs": commit_SHAs,
    }


def get_file_contents(file_path: str) -> str:
    """Useful for getting the contents of a file in the repository given its path,
    for example 'app/models.py'."""
    return _repo.get_contents(file_path).decoded_content.decode("utf-8")


def get_commit_details(head_sha: str) -> list[dict[str, Any]]:
    """Useful for getting the files changed in a commit given its SHA.
    Returns the filename, status, additions, deletions, changes and patch (diff) of each file."""
    commit = _repo.get_commit(head_sha)
    changed_files: list[dict[str, Any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
    return changed_files


def post_review_to_github(pr_number: int, comment: str) -> str:
    """Useful for posting the final review comment on a pull request given the PR number and the comment text."""
    pull_request = _repo.get_pull(pr_number)
    # event="COMMENT" submits the review right away; without it GitHub leaves it as a pending draft
    review = pull_request.create_review(body=comment, event="COMMENT")
    return f"Review posted to PR #{pr_number}: {review.html_url}"


pr_details_tool = FunctionTool.from_defaults(get_pr_details)
file_contents_tool = FunctionTool.from_defaults(get_file_contents)
commit_details_tool = FunctionTool.from_defaults(get_commit_details)
post_review_tool = FunctionTool.from_defaults(post_review_to_github)


async def add_context_to_state(ctx: Context, gathered_contexts: str) -> str:
    """Useful for saving the gathered context (PR details, changed files, requested files) to the state."""
    current_state = await ctx.store.get("state")
    current_state["gathered_contexts"] = gathered_contexts
    await ctx.store.set("state", current_state)
    return "State updated with gathered contexts."


async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Useful for saving the drafted PR review comment to the state."""
    current_state = await ctx.store.get("state")
    current_state["draft_comment"] = draft_comment
    await ctx.store.set("state", current_state)
    return "State updated with draft comment."


async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Useful for saving the final, approved review comment to the state."""
    current_state = await ctx.store.get("state")
    current_state["final_review"] = final_review
    await ctx.store.set("state", current_state)
    return "State updated with final review."


add_context_tool = FunctionTool.from_defaults(add_context_to_state)
add_comment_tool = FunctionTool.from_defaults(add_comment_to_state)

context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers all the needed context from the GitHub repository: PR details, changed files "
                "and any requested files, and saves it to the state.",
    tools=[pr_details_tool, file_contents_tool, commit_details_tool, add_context_tool],
    system_prompt="""You are the context gathering agent. When gathering context, you MUST gather \n:
  - The details: author, title, body, diff_url, state, and head_sha; \n
  - Changed files; \n
  - Any requested for files; \n
Once you gather the requested info, you MUST hand control back to the Commentor Agent.
Always finish in this order: (1) call add_context_to_state with a summary of everything you gathered, then (2) call the handoff tool with to_agent "CommentorAgent". Never end your turn without doing both.""",
    can_handoff_to=["CommentorAgent"],
)

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment comment.",
    tools=[add_comment_tool],
    system_prompt="""You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n
Ensure to do the following for a thorough review:
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent.
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
    - What is good about the PR? \n
    - Did the author follow ALL contribution rules? What is missing? \n
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
    - Are new endpoints documented? - use the diff to determine this. \n
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n
 - If you need any additional details, you must hand off to the Commentor Agent. \n
 - You must hand off to the ReviewAndPostingAgent once you are done drafting a review. \n
 - You should directly address the author. So your comments should sound like: \n
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?" """,
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
)

review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Reviews the draft comment written by the CommentorAgent, requests rewrites if needed, "
                "saves the final review to the state and posts it to GitHub.",
    tools=[add_final_review_to_state, post_review_tool],
    system_prompt="""You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
 When you are satisfied, post the review to GitHub.
 Always finish in this order: (1) call add_final_review_to_state with the final review text, then (2) call post_review_to_github with the PR number and the review text.""",
    can_handoff_to=["CommentorAgent"],
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review": "",
    },
)


async def main():
    query = f"Write a review for PR number {pr_number} and post it to GitHub. Remember, the PR number is coming from the environment."
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    git.close()
