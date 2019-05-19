# coding=UTF-8
import pygame
from pygame.sprite import Sprite

class Alien(Sprite):
    def __init__(self,ai_settings,screen):
        super().__init__()
        self.ai_settings=ai_settings
        self.screen=screen
        #获取图像
        self.image=pygame.image.load('/Users/apple/Desktop/alien_invasion/image/alien.bmp')
        self.rect=self.image.get_rect()
        #外星人最初显示在屏幕左上角附近
        self.rect.x=self.rect.width
        self.rect.y=self.rect.height
        #外星人准确位置，浮点数
        self.x=float(self.rect.x)

    def blitme(self):
        self.screen.blit(self.image,self.rect)

    def check_edges(self):
        screen_rect=self.screen.get_rect()
        if self.rect.right >= screen_rect.right:
            return True
        elif self.rect.left <= 0:
            return True

    def update(self):#向右移动外星人
        self.x+=(self.ai_settings.alien_speed_factor*
                 self.ai_settings.fleet_direction)
        self.rect.x=self.x
