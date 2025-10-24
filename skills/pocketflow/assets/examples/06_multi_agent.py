"""
PocketFlow Cookbook Example: Multi-Agent (Taboo Game)

Difficulty: ★☆☆ Beginner Level
Source: https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-multi-agent

Description:
Two agents playing Taboo word game with async communication.
Demonstrates:
- Multi-agent systems
- Async message queues for inter-agent communication
- AsyncNode and AsyncFlow
- Self-looping async nodes
- Game logic with termination conditions
"""

import asyncio
from pocketflow import AsyncNode, AsyncFlow
# from utils import call_llm  # You need to implement this


class AsyncHinter(AsyncNode):
    """Agent that provides hints for the target word"""

    async def prep_async(self, shared):
        """Wait for guess from guesser"""
        guess = await shared["hinter_queue"].get()

        if guess == "GAME_OVER":
            return None

        return (
            shared["target_word"],
            shared["forbidden_words"],
            shared.get("past_guesses", [])
        )

    async def exec_async(self, inputs):
        """Generate hint avoiding forbidden words"""
        if inputs is None:
            return None

        target, forbidden, past_guesses = inputs

        prompt = f"Generate hint for '{target}'\nForbidden words: {forbidden}"
        if past_guesses:
            prompt += f"\nPrevious wrong guesses: {past_guesses}\nMake hint more specific."
        prompt += "\nUse at most 5 words."

        # hint = call_llm(prompt)
        hint = "Thinking of childhood summer days"  # Placeholder

        print(f"\nHinter: Here's your hint - {hint}")
        return hint

    async def post_async(self, shared, prep_res, exec_res):
        """Send hint to guesser"""
        if exec_res is None:
            return "end"

        # Send hint to guesser's queue
        await shared["guesser_queue"].put(exec_res)
        return "continue"


class AsyncGuesser(AsyncNode):
    """Agent that guesses the target word from hints"""

    async def prep_async(self, shared):
        """Wait for hint from hinter"""
        hint = await shared["guesser_queue"].get()
        return hint, shared.get("past_guesses", [])

    async def exec_async(self, inputs):
        """Make a guess based on hint"""
        hint, past_guesses = inputs

        prompt = f"""
Given hint: {hint}
Past wrong guesses: {past_guesses}
Make a new guess. Reply with a single word:
"""
        # guess = call_llm(prompt)
        guess = "memories"  # Placeholder

        print(f"Guesser: I guess it's - {guess}")
        return guess

    async def post_async(self, shared, prep_res, exec_res):
        """Check guess and update game state"""
        # Check if correct
        if exec_res.lower() == shared["target_word"].lower():
            print("\n✅ Game Over - Correct guess!")
            await shared["hinter_queue"].put("GAME_OVER")
            return "end"

        # Store wrong guess
        if "past_guesses" not in shared:
            shared["past_guesses"] = []
        shared["past_guesses"].append(exec_res)

        # Send guess to hinter
        await shared["hinter_queue"].put(exec_res)
        return "continue"


async def main():
    """Run the Taboo game"""

    # Game setup
    shared = {
        "target_word": "nostalgia",
        "forbidden_words": ["memory", "past", "remember", "feeling", "longing"],
        "hinter_queue": asyncio.Queue(),
        "guesser_queue": asyncio.Queue()
    }

    print("\n" + "="*50)
    print("🎮 Taboo Game Starting!")
    print("="*50)
    print(f"Target word: {shared['target_word']}")
    print(f"Forbidden words: {shared['forbidden_words']}")
    print("="*50 + "\n")

    # Initialize game with empty guess
    await shared["hinter_queue"].put("")

    # Create agents
    hinter = AsyncHinter()
    guesser = AsyncGuesser()

    # Setup self-loops
    hinter - "continue" >> hinter
    guesser - "continue" >> guesser

    # Create flows
    hinter_flow = AsyncFlow(start=hinter)
    guesser_flow = AsyncFlow(start=guesser)

    # Run both agents concurrently
    await asyncio.gather(
        hinter_flow.run_async(shared),
        guesser_flow.run_async(shared)
    )

    print("\n" + "="*50)
    print("🏁 Game Complete!")
    print(f"Total guesses: {len(shared.get('past_guesses', []))}")
    print("="*50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
