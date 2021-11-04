# Some guidlines

1. DL projects on robot arm from [Harvard](https://www.youtube.com/watch?v=VO1mCjHvzlo), we can refer their video and methods

2. LiHongyi machine learning [courses](https://www.bilibili.com/video/BV1zA411K7en?p=9) + [reinforcement learning](https://www.bilibili.com/video/av24724071/?p=4)

3. mujoco module built [a](https://zhuanlan.zhihu.com/p/99991106), [b](https://zhuanlan.zhihu.com/p/143983506), [c](https://zhuanlan.zhihu.com/p/99991106)

4. [morvan robot arm](https://mofanpy.com/tutorials/machine-learning/ML-practice/), we can learn his pyglet programming code



## Debugs on environment

1. Try to hit the ball into wall （waiting keyboard）
2. see the new velocity_n after solving equation (2 solutions)
3. when hitting the wall, the velocity change the orientation, but maybe the original orientation is corrent, the changing one has problem due to integral 

## Problems

1. pyglet load picture

   there are two ways,

- pyglet.image.load("XXX.png")   : working directory
- pyglet.resource.image("xxx.png")  : a path relative to the script

