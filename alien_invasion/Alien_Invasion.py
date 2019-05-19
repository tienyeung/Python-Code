#!/usr/bin/env python3
#coding=UTF-8
import pygame
from pygame.sprite import Group

import game_function
from setting import Settings
from ship import Ship
from game_stats import GameStats
from button import Button
from scoreboard import Scoreboard

def run_game():
    pygame.init()#初始化
    ai_settings=Settings()#实例化
    screen=pygame.display.set_mode((ai_settings.screen_width,ai_settings.screen_height))#设置屏幕尺寸
    pygame.display.set_caption('Alien Invasion')
    #创建play按钮
    play_button=Button(ai_settings,screen,'PLAY')
    #创建用于存储统计信息的实例以及记分牌
    stats=GameStats(ai_settings)
    sb=Scoreboard(ai_settings,screen,stats)
    #创建飞船，子弹编组，外星人编组
    ship=Ship(ai_settings,screen)
    bullets=Group()
    aliens=Group()
    #创建外星人群
    game_function.create_fleet(ai_settings,screen,ship,aliens)



    while True:
        game_function.check_events(ai_settings,screen,stats,sb,
                                   play_button,ship,aliens,bullets)#监听键盘事件

        if stats.game_active:
            ship.update()#更新飞船位置
        #更新子弹位置并删除超出屏幕的子弹
            game_function.update_bullets(ai_settings,screen,stats,sb,
                                         ship,aliens,bullets)
            game_function.update_aliens(ai_settings,stats,sb,
                                    screen,ship,aliens,bullets)#更新外星人位置
        game_function.update_screen(ai_settings,screen,stats,sb,ship,
                                    aliens,bullets,play_button)#更新屏幕


run_game()
