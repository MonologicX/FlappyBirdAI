import pygame

YELLOW = (255, 255, 0)

class Bird:
    def __init__(self, FPS, WINHEIGHT, startPos, size, flapForce=10, maxSpeed=10, gravity=9.81, maxGravity=20):

        self.hitbox = pygame.Rect(startPos, size)
        self.gravity = gravity
        self.maxGravity = maxGravity
        self.FPS = FPS
        self.flapForce = flapForce
        self.maxSpeed = maxSpeed

        self.WINHEIGHT = WINHEIGHT

        self.yVelocity = 0

    def move(self, action):

        if action == 0:
            return
        elif action == 1:
            self.yVelocity = 0
            self.yVelocity = -self.flapForce

    def update(self, WIN):
        
        #Gravity
        if self.yVelocity < self.maxGravity:
            self.yVelocity += self.gravity
        else:
            self.yVelocity = self.maxGravity
        
        self.hitbox.y += self.yVelocity

        if self.hitbox.y < 0:
            self.hitbox.y -= self.yVelocity
            self.hitbox.y = 0
        elif (self.hitbox.y + self.hitbox.height) > self.WINHEIGHT:
            self.hitbox.y -= self.yVelocity
            self.hitbox.y = self.WINHEIGHT - self.hitbox.height

        self.draw(WIN)
    
    def draw(self, WIN):
        pygame.draw.rect(WIN, YELLOW, self.hitbox)