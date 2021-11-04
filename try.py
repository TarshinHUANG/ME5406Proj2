import pyglet

window = pyglet.window.Window()
image1 = pyglet.resource.image('robot.png')

#image1 = pyglet.image.load('robot.png')
image1.width=100
image1.height=100
image1.anchor_x = image1.width // 2
image1.anchor_y = image1.height // 2
batch = pyglet.graphics.Batch()
ball=[]
def render():
    pyglet.gl.glClearColor(0.5,0.5,0.5,0.5)  # the windows color
    #window.switch_to()
    robot=pyglet.sprite.Sprite(image1, x=100, y=100,batch=batch)
    robot.rotation=270
    ball.append(robot)
    robot=pyglet.sprite.Sprite(image1, x=200, y=100,batch=batch)
    ball.append(robot)
    
    window.dispatch_events()
    window.clear()
    #image.blit(0, 0)
    #image.blit(100,100)
    batch.draw()
    window.flip()

if __name__=='__main__':
    render()