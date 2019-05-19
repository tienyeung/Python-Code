# coding=UTF-8
import pygame
from pygame.sprite import Sprite

class Ship(Sprite):
    def __init__(self,ai_settings,screen):
        super().__init__()
        self.screen=screen
        self.ai_settings=ai_settings
        #加载图形并获取外接矩形
        self.image=pygame.image.load('/Users/apple/Desktop/alien_invasion/image/ship.bmp')
        self.rect=self.image.get_rect()
        self.screen_rect=screen.get_rect()

        #飞船放在屏幕下中央
        self.rect.centerx=self.screen_rect.centerx
        self.rect.bottom=self.screen_rect.bottom

        #飞船中央化为浮点数
        self.center=float(self.rect.centerx)

        #移动标志
        self.moving_right=False
        self.moving_left=False

    def update(self):
        if self.moving_right and self.rect.right<self.screen_rect.right:
            self.center += self.ai_settings.ship_speed_factor
        if self.moving_left and self.rect.left>0:
            self.center -= self.ai_settings.ship_speed_factor

        self.rect.centerx=self.center

    def blitme(self):#绘制位图：screen.blit(位图，rect)
        self.screen.blit(self.image,self.rect)

    def center_ship(self):
        self.center=self.screen_rect.centerx
