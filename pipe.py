import pygame
import random

class PipeSet:
    def __init__(self, WINSIZE, scrollSpeed=5, pipeWidth=80, gapSize=200):
        self.WINWIDTH = WINSIZE[0]
        self.WINHEIGHT = WINSIZE[1]
        self.pipeWidth = pipeWidth
        self.gapSize = gapSize
        self.scrollSpeed = scrollSpeed

        self.top = pygame.Rect((self.WINWIDTH + self.pipeWidth, 0), (self.pipeWidth, random.randint(0, (self.WINHEIGHT - self.gapSize))))
        self.bot = pygame.Rect((self.top.x, self.top.height + self.gapSize), (self.pipeWidth, (self.WINHEIGHT - (self.top.height + self.gapSize))))
        self.gap = pygame.Rect((self.top.x, self.top.height), (self.pipeWidth, self.gapSize))

    def scroll(self):
        self.top.x -= self.scrollSpeed
        self.bot.x -= self.scrollSpeed
        self.gap.x -= self.scrollSpeed

    def draw(self, WIN):
        pygame.draw.rect(WIN, (255, 255, 255), self.top)
        pygame.draw.rect(WIN, (255, 255, 255), self.bot)
        pygame.draw.rect(WIN, (50, 50, 50), self.gap)

    def update(self, WIN):
        self.scroll()
        self.draw(WIN)
    
    def collectPoint(self):
        self.gap.y += self.gap.height / 2
        self.gap.size = (0, 0)
        return 1