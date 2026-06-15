"""
Local Orbit Wars match runner.

Plays your agent (main.py) against the built-in "random" opponent, prints the
result, and writes an HTML replay you can open in your browser.

Run it with:   python run.py
Then open:     replay.html
"""

from kaggle_environments import make

# debug=True surfaces errors from your agent instead of silently failing.
env = make("orbit_wars", configuration={"seed": 42}, debug=True)

# Load main.py the same way Kaggle does when you submit, and play it against
# the built-in random agent. Swap "random" for "main.py" to play yourself.
env.run(["main.py", "random"])

# Final step holds each player's reward (1 win / 0 loss / etc.) and status.
final = env.steps[-1]
for i, state in enumerate(final):
    print(f"Player {i}: reward={state.reward}, status={state.status}")

# Save a watchable replay. (mode="ipython" is for notebooks; "html" for files.)
html = env.render(mode="html", width=800, height=600)
with open("replay.html", "w") as f:
    f.write(html)
print("Wrote replay.html — open it in your browser to watch the match.")
