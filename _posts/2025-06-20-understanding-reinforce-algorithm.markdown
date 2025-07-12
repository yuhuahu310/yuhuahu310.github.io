---
layout: single
title:  "Understanding REINFORCE"
categories: jekyll update
author_profile: true
comments: true
share: true
toc: true
toc_label: "Contents"
toc_sticky: true
---

Some RL learning notes.

## What Are Policy Gradient Methods?
This is one class of RL algorithms that learns a parameterized policy. It directly learns the policy itself that gives the action probability based on current state.

In constrast, some RL algorithms first learn a action-value function and then select actions based on this value function.

In simple math terms, it learns $\pi_{\theta}(s)$ (Imagine a neural network parameterized by $\theta$, given input state $s$, can output a probability for each eligible action. This action can be discrete or continuous).

## Policy Gradient Theorem
We want to maximize $J(\theta) = v_{\pi}(s_0)$, the value at state $s_0$ if we follow policy $\pi_{\theta}$. We can take derivatives with respect to $\theta$ and update it, but changing $\theta$ affects two
things: 

$$
\begin{align}
1. &\quad \pi_{\theta}(a | s) \\
2. &\quad \text{The distribution of states visited under } \pi_{\theta}. 
\end{align}
$$

It is often impossible to differentiate through the state distribution because it depends on unknown environment factors.
Updating the policy affects what actions are selected, and the action then determines how the environment transitions to the next state, according to the transition probability function
$P(s^{\prime}|s, a)$
which is specific to each environment. So we want to avoid differentiating through it.

The policy gradient theorem helps with exactly that. It states that

$$
\begin{align}
\nabla_{\theta} J(\theta) &\propto \sum_{s} \mu(s) \sum_{a} q_{\pi}(s, a) \nabla \pi(a|s, \theta) \\
&= \mathbb{E}_{\pi} \left[ \sum_{a} q_{\pi}(S, a) \nabla \pi(a|S, \theta) \right]
\end{align}
$$

where

$q_{\pi}(s, a)$ is the action-value function under policy $\pi_{\theta}$

$\mu(s)$ is the stationary state distribution under policy $\pi_{\theta}$

S is capitalized to show that it is a random variable

Note how we only have to differentiate through $\pi$ which we fully know how to do. The other values can be estimated using various methods, e.g. Monte Carlo.

I highly recommend Lilian Weng's [blog post](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#policy-gradient-theorem) for understanding the proof of this theorem. It has detailed step-by-step derivation.

## REINFORCE algorithm

REINFORCE is a basic application of the policy gradient theorem.

To get the REINFORCE update we need a few more steps:

$$
\begin{align}
\nabla_{\theta} J(\theta) &= \mathbb{E}_{\pi} \left[ \sum_{a} q_{\pi}(S, a) \nabla \pi(a|S, \theta) \right] \\
&= \mathbb{E}_{\pi} \left[ \sum_{a} \pi(a|S, \theta) q_{\pi}(S, a) \frac{\nabla \pi(a|S, \theta)}{\pi(a|S, \theta)} \right] \\
&= \mathbb{E}_{\pi} \left[ q_{\pi}(S, A) \nabla \log \pi(A|S, \theta) \right]
\end{align}
$$

Here the random variables $S$ and $A$ are a bit tricky to interpret.

The distribution of $S$ is the stationary distribution of states if we follow the policy $\pi$.

The distribution of $A$ means how likely we are selecting each action according to policy $\pi$.

We just need a fairly accurate estimate of these quantities to compute the gradient.

Here's how the algorithm goes:

<div class=algorithm>

<ol type="1">

<li><b>Input:</b> Differentiable policy $\pi(a|s, \theta)$ initialized to $\theta$. Step size $\alpha$. Discounting factor $\gamma$.</li>
<li>while (True) <b>do</b>: </li>
        <div id="self-play-for-loop">
<li>    Generate an episode $s_0, a_0, r_1, ..., s_{T-1}, a_{T-1}, r_T$ </li>
<li>    Loop for each time step $t = 0, 1, ..., T-1$: </li>
          <div id="self-play-second-for-loop">
<li>      $G = \sum_{k=t+1}^{T} \gamma^{k-t-1} r_k$ </li>
<li>      $\theta = \theta + \alpha \gamma^t G \nabla \log \pi(a_t|s_t, \theta)$ </li>
          </div><!-- self-play-second-for-loop -->
        </div><!-- self-play-for-loop -->
</ol>

</div><!-- algorithm -->

$G$ estimates $q_{\pi}(S, A)$ by summing over all future discounted rewards in this episode.

The update also includes the step size and discounting factor that exponentially decreases with $t$.

## References

[1] https://yugeten.github.io/posts/2025/01/ppogrpo/

[2] https://lilianweng.github.io/posts/2018-02-19-rl-overview/#policy-gradient 

[3] Sutton and Barto (2020). Reinforcement Learning: An Introduction.

[4] https://en.wikipedia.org/wiki/Policy_gradient_method

[5] Proximal Policy Optimization Algorithms: https://arxiv.org/abs/1707.06347
