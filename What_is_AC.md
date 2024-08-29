## What is AC?

AC, short for Andrews-Curtis, is an old and still unsolved conjecture in Math. It’s about finding incredibly rare paths in a vast, infinite search space. Let’s first break down the search space using Python objects and operations, and then we’ll connect it back to the Math terms.

In this space, an element is a pair of tuples, where each element in the tuple can be 1, -1, 2, or -2.[^1] Two elements are considered connected if they’re related by a sequence of the following actions:

- **Concatenation:** You can concatenate one tuple with another. For example, `[(1, 2, -1), (2, 1)]` is connected to `[(1, 2, -1), (2, 1, 1, 2, -1)]` and `[(1, 2, -1, 2, 1), (2, 1)]`.

- **Conjugation:** You can prepend a tuple with any of the four elements (1, -1, 2, -2), while also appending its negative at the end. For instance, `(1, 2, -1)` can be changed to `(-2, 1, 2, -1, 2)`.

- **Inversion:** You can replace a tuple with its reverse and negative version. For example, `(2, 1, 2, -1)` could be replaced with `(1, -2, -1, -2)`.

And that’s it! These are all the actions you can take, often called "AC moves."

For those familiar with reinforcement learning, this problem presents a perfect playground for RL! The observation space is the set of tuples, and the action space consists of the moves described above.

So, what’s the conjecture? It proposes that there’s a path between every element in this space and the "trivial state" `[(1,), (2,)]`. This conjecture has piqued the interest of Mathematicians for decades, especially those working in combinatorial group theory and algebraic topology.

## Why AC is useful for RL?

But why should the reinforcement learning community care about AC? This environment offers a unique mix of challenges, which we believe are essential for pushing RL agents' capabilities toward AGI:

- **Sparse rewards:** Since the final state has the shortest tuples (with the sum of lengths being 2), a good candidate for the reward function is simply the negative of the total sum of lengths. However, a high positive reward is only given when the final state is reached, making the reward function extremely sparse.

- **Control over horizon length:** While many reinforcement learning environments allow users to vary horizon length (e.g., by using Gymnasium’s TimeLimit Wrapper), not all of them offer the same level of flexibility as AC.

- **Wide range of initial state complexities:** In AC environments, some states are significantly easier to solve than others. In fact, the hardest states may require paths of millions, or even billions, of steps.[^2] This is far beyond the length of a chess or Go game. Developing RL algorithms that can handle problems of this scale is a crucial step toward AGI.

- **Low computational cost:** Each action is just an operation on two tuples, making this environment computationally cheaper than many other well-known RL environments like Atari, MuJoCo, or ProcGen.

## Implementation of ACEnv

In this repository, we’ve implemented the AC environment using two classes: `ACEnvConfig` and `ACEnv`. The former is a dataclass with attributes `initial_state` and `horizon_length`. `initial_state` is a Python list or a NumPy array of even length, where each half is one of the two tuples padded with zeros on the right. For example, the state `[(1, 2, -1), (2, -1)]` is represented as `[1, 2, -1, 0, 0, 2, -1, 0, 0, 0]`.

In mathematical terms, each of the tuples is called a "relator." The number of zeros we pad, and hence, the length of the initial state, is implicitly an important property of `ACEnvConfig`. One half of this length is what we call `max_relator_length` in the code, as any state with a relator longer than this length will be ignored during the search. Placing such restrictions is crucial because, without them, the search space would simply be infinite!

`ACEnv` is a Gymnasium environment that you can import and train with any popular RL libraries, such as Stable Baselines or RLlib. For full disclosure, its actions aren’t exactly the same as the AC moves described above. Instead, its actions are sequences of AC moves themselves. We made this adjustment because the inversion move doesn’t change the lengths of tuples, and therefore, doesn’t provide any signals to RL agents. Mixing it with other moves turned out to be essential in speeding up the training process.

## Math Terminology and Notation

Finally, let’s discuss how these states are often presented in Math. In Math, 1, 2, -1, and -2 are often written as $x$, $y$, $x^{-1}$, and $y^{-1}$. A tuple is written as a word in these letters. For example, `(1, 2, -1)` is written as $xyx^{-1}$. $x^{-1}$ and $y^{-1}$ are inverses of $x$ and $y$, respectively, so $x x^{-1}$ and $y y^{-1}$ just cancel out. A pair of tuples is often written using angle brackets, so that `[(1, 2, -1), (2, 1)]` is written as $\langle x y x^{-1}, y x \rangle$ or sometimes $\langle x, y | x y x^{-1}, yx \rangle$ to signify that the relators are words in $x$ and $y$ (and their inverses) only. The conjecture essentially states that you can relate any of these presentations to $\langle x, y \rangle$ through a sequence of AC moves.

[^1]: Not every pair of tuples is actually allowed, and that’s a big challenge in itself. But we’ll leave that detail aside for now.

[^2]: Such states are known to exist, provably so.