#coding=UTF-8
import pygame
from pygame.sprite import Sprite
#精灵可以认为成是一个个小图片，一种可以在屏幕上移动的图形对象，并且可以与其他图形对象交互
class Bullet(Sprite):
    def __init__(self,ai_settings,screen,ship):
        super().__init__()#继承超类
        self.screen=screen

        #(0,0)创建子弹，并设置正确位置
        self.rect=pygame.Rect(0,0,ai_settings.bullet_width,
                              ai_settings.bullet_height)
        self.rect.centerx=ship.rect.centerx
        self.rect.top=ship.rect.top
        self.y=float(self.rect.y)
        #子弹位置的小数值
        self.color=ai_settings.bullet_color
        self.speed_factor=ai_settings.bullet_speed_factor

    def update(self):
        """向上移动子弹"""
        self.y -= self.speed_factor
        self.rect.y=self.y

    def draw_bullet(self):
        """屏幕上绘制子弹"""
        pygame.draw.rect(self.screen,self.color,self.rect)
