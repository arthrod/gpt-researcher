import asyncio
<<<<<<< HEAD

=======
from gpt_researcher.utils.llm import get_llm
>>>>>>> 1027e1d0 (Fix linting issues)
from dotenv import load_dotenv

from gpt_researcher.utils.llm import get_llm

load_dotenv()


async def main():
    # Example usage of get_llm function
    """
    Asynchronously create an LLM client with a sample configuration, print its settings, and run a quick runtime test via test_llm.
    
    This function demonstrates usage of get_llm by constructing an OpenAI-based client (model "gpt-3.5-turbo", temperature 0.7, max_tokens 1000), printing the chosen configuration and the returned client object, then awaiting test_llm(llm) to verify the client can produce a chat response.
    """
    llm_provider = "openai"
    model = "gpt-3.5-turbo"
    temperature = 0.7
    max_tokens = 1000

    llm = get_llm(
        llm_provider, model=model, temperature=temperature, max_tokens=max_tokens
    )
    print(
        f"LLM Provider: {llm_provider}, Model: {model}, Temperature: {temperature}, Max Tokens: {max_tokens}"
    )
    print("llm: ", llm)
    await test_llm(llm=llm)


async def test_llm(llm):
    # Test the connection with a simple query
    messages = [{"role": "user", "content": "sup?"}]
    try:
        response = await llm.get_chat_response(messages, stream=False)
        print("LLM response:", response)
    except Exception as e:
        print(f"Error: {e}")


# Run the async function
asyncio.run(main())
