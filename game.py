import pygame
import sys
from bird import Bird
from pipe import PipeSet

class FlappyBird:
    def __init__(self, FPS=30, bFlapForce=30, bGravity=4, bMaxGravity=10, scrollSpeed=5):
        pygame.init()
        self.WINWIDTH, self.WINHEIGHT = (1000, 1000)
        self.WIN = pygame.display.set_mode((self.WINWIDTH, self.WINHEIGHT))
        self.CLOCK = pygame.time.Clock()
        self.FPS = FPS

        self.birdFlapForce = bFlapForce
        self.birdGravity = bGravity
        self.birdMaxGravity = bMaxGravity
        self.scrollSpeed = scrollSpeed

        self.reset()
    
    def reset(self):
        self.score = 0
        self.gameOver = False

        self.bird = Bird(self.FPS, self.WINHEIGHT, ((self.WINWIDTH / 5), (self.WINHEIGHT / 2)), (20, 20), flapForce=self.birdFlapForce, gravity=self.birdGravity, maxGravity=self.birdMaxGravity)
        self.pipes = PipeSet((self.WINWIDTH, self.WINHEIGHT), scrollSpeed=self.scrollSpeed)

    def gameStep(self, action):

        reward = 1

        self.bird.move(action)

        self.update()

        if self.pipes.top.x < -self.pipes.pipeWidth:
            self.pipes = PipeSet((self.WINWIDTH, self.WINHEIGHT), scrollSpeed=self.scrollSpeed)
        
        if self.pipes.gap.colliderect(self.bird.hitbox):
            self.score += self.pipes.collectPoint()
            reward += 1000
        
        reward += 10 - abs(self.bird.hitbox.centery - self.pipes.gap.centery) / 1000
        
        
        if self.bird.hitbox.colliderect(self.pipes.top) or self.bird.hitbox.colliderect(self.pipes.bot):
            reward = -100
            self.gameOver = True

        return reward, self.gameOver

    def update(self):
        self.WIN.fill((0, 0, 0))
        self.pipes.update(self.WIN)
        self.bird.update(self.WIN)

        pygame.display.update()
        self.CLOCK.tick(self.FPS)
    
    def humanTest(self):
        while True:
            while not self.gameOver:

                move = 0

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            move = 1
        
                reward, gameOver = self.gameStep(move)
                print(reward)
            
            while self.gameOver:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            self.reset()
    
    def getState(self):

        xDistanceToPipe = float(self.bird.hitbox.centerx - self.pipes.gap.x)
        yDistanceToGap = float(self.bird.hitbox.centery - self.pipes.gap.centery)
        birdVelocity = float(self.bird.yVelocity)
        birdY = float(self.bird.hitbox.y)
        gapTop = float(self.pipes.gap.top)
        gapBot = float(self.pipes.gap.bottom)

        stateTensor = [xDistanceToPipe, yDistanceToGap, birdVelocity, birdY, gapTop, gapBot]
        return stateTensor
    
    def trainStep(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        return self.gameStep(action)

#g = FlappyBird()
#g.humanTest()