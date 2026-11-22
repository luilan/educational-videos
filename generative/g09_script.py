"""How Generative Models Work: Images and Sound, episode 9 — Flow Matching. One entry per narration section.

Every number comes from code/g09_flow_matching (2026-11-22): episode 6's U-Net trained with flow matching on the same
data for the same 20,000 steps; 500 samples from the same starting noise, scored by episode 3's judge; plus a 2D
spiral with diffusion, flow matching and reflow.
"""
TITLE = "Flow Matching"
TAGLINE = "A straighter road from noise to picture."
NEXT = "Conditioning: Telling the Model What to Draw"
SECTIONS = [
    # 1. Hook
    "Five network calls. Episode six's diffusion model, with its best sampler: seven percent of its pictures are "
    "recognizable. The same network, trained a different way: seventy-six percent.",
    # 2. The idea
    "That way is flow matching. Make the training input by sliding along a straight line: pure noise at time one, "
    "the picture at time zero, and a simple mix in between. The network learns the direction of that line, noise "
    "minus picture. To generate, start from noise and step against the predicted direction.",
    # 3. Versus diffusion
    "Diffusion's mix follows a curved schedule, and its network predicts only the noise. Here the target is the "
    "whole direction of travel, and a straight line can be crossed in a few long steps.",
    # 4. Toy
    "On the spiral, with four steps, flow matching already sits on the curve, while diffusion is still spread out. "
    "With a single step, though, flow matching puts every point in the same place: the middle.",
    # 5. Not straight
    "Because the learned paths aren't straight. In training, lines from many noises to many points cross each other, "
    "and where they cross, the network can only learn their average direction. So the paths bend, and one step "
    "follows the average all the way to the middle.",
    # 6. Reflow
    "The fix is called reflow. Run the trained flow, pair each noise with the point it lands on, and train a new flow "
    "on just those pairs. Their lines don't cross, so the paths come out straight, and now a single step lands on the "
    "spiral.",
    # 7. Shapes
    "Back to Shapes, the same U-Net with the same training. With five steps: seventy-six percent against seven. With "
    "ten: ninety-one against eighty-five. From twenty steps on, diffusion catches up, and both are in the nineties.",
    # 8. In practice
    "That's the trade: good pictures in far fewer steps, for the same training. It's why recent image models such "
    "as Stable Diffusion 3 and Flux are trained with flow matching.",
    # 9. Code
    "In code: mix on a straight line, predict noise minus picture, and step back along it.",
    # 10. Next
    "Next: conditioning, how to tell the model what to draw.",
]
