# Generative Adversarial Networks

In this article, we will see how Generative Adversarial Networks (GANs) [1]  work, and how we can use them to generate realistic images.

# 1 - Intro to Generative Adversarial Networks

Generative Adversarial Networks (GANs) were introduced by Ian Goodfellow (2014) as "a new framework for estimating generative models via an adversarial process, in which two models , discriminator and generator, are trained simultaneously."

The discriminator (D) is trained to distinguish between real data samples, drawn from the true data distribution, and fake samples produced by the generator (G).

Through this adversarial process, both models are expected to improve over time: the Discriminator becomes better at detecting fakes, while the Generator learns to produce increasingly realistic images (in other words, the fool ends up being fooled).

# 2 - Underlying theory of GANs

Let $x$ be a sample from the real data distribution, such that $x \sim p_{\text{data}}(x)$, where $p_{\text{data}}(x)$ represents the unknown distribution of the real data. 

In probabilistic terms, we want to learn a generative model $p_{\text{\theta}}(x)$, such that $\theta$ is optimized to approximate the true data distribution $p_{\text{\theta}}(x) \sim p_{\text{data}}(x)$.

