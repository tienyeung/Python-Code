# coding=UTF-8
import sys
import pygame
from time import sleep

from bullet import Bullet
from alien import Alien

def check_keydown_events(event,ai_settings,screen,ship,bullets):
    if event.key==pygame.K_RIGHT:#右
        ship.moving_right=True
    elif event.key==pygame.K_LEFT:#左
        ship.moving_left=True
    elif event.key==pygame.K_SPACE:#空格
        fire_bullet(ai_settings,screen,ship,bullets)
    elif event.key==pygame.K_q:
        sys.exit()

def fire_bullet(ai_settings,screen,ship,bullets):#若未到达限制，则发射子弹
    if len(bullets) < ai_settings.bullets_allowed:
        new_bullet=Bullet(ai_settings,screen,ship)
        bullets.add(new_bullet)


def check_upkey_events(event,ship):
    if event.key==pygame.K_RIGHT:
        ship.moving_right=False
    elif event.key==pygame.K_LEFT:
        ship.moving_left=False

def check_events(ai_settings,screen,stats,sb,play_button,ship,aliens,bullets):
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            sys.exit()
        elif event.type==pygame.KEYDOWN:#监听触发方向键事件
            check_keydown_events(event,ai_settings,screen,ship,bullets)
        elif event.type==pygame.MOUSEBUTTONDOWN:
            mouse_x,mouse_y=pygame.mouse.get_pos()
            check_play_button(ai_settings,screen,stats,sb,play_button,
                              ship,aliens,bullets,mouse_x,mouse_y)
        elif event.type==pygame.KEYUP:#松开方向键
            check_upkey_events(event,ship)

def check_play_button(ai_settings,screen,stats,sb,play_button,
                      ship,aliens,bullets,mouse_x,mouse_y):
    #单击play开始游戏
    button_clicked=play_button.rect.collidepoint(mouse_x,mouse_y)
    if button_clicked and not stats.game_active:
        #重置游戏设置
        ai_settings.initialize_dynamic_settings()
        #隐藏光标
        pygame.mouse.set_visible(False)
        #重置游戏
        stats.reset_stats()
        stats.game_active=True
        #重置记分牌
        sb.prep_score()
        sb.prep_high_score()
        sb.prep_level()
        sb.prep_ships()

        #清空外星人和子弹
        aliens.empty()
        bullets.empty()
        #创建新外星人和子弹，并让飞船居中
        create_fleet(ai_settings,screen,ship,aliens)
        ship.center_ship()

def update_screen(ai_settings,screen,stats,sb,ship,aliens,bullets,play_button):
    screen.fill(ai_settings.bg_color)#刷新屏幕颜色
    #重绘所有子弹
    for bullet in bullets.sprites():
        bullet.draw_bullet()
    ship.blitme()#绘制图形
    aliens.draw(screen)
    #显示得分
    sb.show_score()
    #绘制play按钮
    if not stats.game_active:
        play_button.draw_button()

    pygame.display.flip()#刷新

def update_bullets(ai_settings,screen,stats,sb,ship,aliens,bullets):
    bullets.update()
    #删除子弹
    for bullet in bullets.copy():
        if bullet.rect.bottom <= 0:
            bullets.remove(bullet)
    check_bullet_alien_collisions(ai_settings,screen,
                                  stats,sb,ship,aliens,bullets)

def check_bullet_alien_collisions(ai_settings,screen,
                                  stats,sb,ship,aliens,bullets):
    #检查子弹是否击中外星人
    #下列函数遍历子弹和外星人编组，若碰撞，则在字典里返回相应键值对
    collisions=pygame.sprite.groupcollide(bullets,aliens,True,True)
    if collisions:
        for aliens in collisions.values():
            stats.score += ai_settings.alien_point*len(aliens)
            sb.prep_score()
        check_high_score(stats,sb)

    if len(aliens) == 0:
        #删除现有子弹并新建外星人
        bullets.empty()
        ai_settings.increase_speed()
        #提升等级
        stats.level+=1
        sb.prep_level()

        create_fleet(ai_settings,screen,ship,aliens)

def get_number_aliens_x(ai_settings,alien_width):
    #创建一群外星人，个数为（屏幕宽度-2个外星人宽度）/（2*外星人宽度）
    available_space_x=ai_settings.screen_width - 2*alien_width
    number_aliens_x=int(available_space_x/(2*alien_width))
    return number_aliens_x

def get_number_rows(ai_settings,ship_height,alien_height):
    available_space_y=(ai_settings.screen_height-
                       (3*alien_height)-ship_height)
    number_rows=int(available_space_y/(2*alien_height))
    return number_rows

def create_alien(ai_settings,screen,aliens,alien_number,row_number):
    alien=Alien(ai_settings,screen)
    alien_width=alien.rect.width
    alien.x=alien_width+2*alien_width*alien_number
    alien.rect.x=alien.x
    alien.rect.y=alien.rect.height+2*alien.rect.height*row_number
    aliens.add(alien)

def create_fleet(ai_settings,screen,ship,aliens):
    alien=Alien(ai_settings,screen)
    number_aliens_x=get_number_aliens_x(ai_settings,alien.rect.width)
    number_rows=get_number_rows(ai_settings,ship.rect.height,alien.rect.height)
    #创建第一行外星人
    for row_number in range(number_rows):
        for alien_number in range(number_aliens_x):
            create_alien(ai_settings,screen,aliens,alien_number,
                         row_number)

def check_fleet_edges(ai_settings,aliens):
    """有外星人到达边缘时采取的措施"""
    for alien in aliens.sprites():
        if alien.check_edges():
            change_fleet_direction(ai_settings,aliens)
            break

def check_aliens_bottom(ai_settings,stats,sb,screen,ship,aliens,bullets):
    """有外星人到达底部时采取的措施"""
    screen_rect=screen.get_rect()
    for alien in aliens.sprites():
        if alien.rect.bottom >= screen_rect.bottom:
            ship_hit(ai_settings,stats,sb,screen,ship,aliens,bullets)
            break

def change_fleet_direction(ai_settings,aliens):
    """整群外星人下移并改变方向"""
    for alien in aliens.sprites():
        alien.rect.y += ai_settings.fleet_drop_speed
        ai_settings.fleet_direction *= -1

def update_aliens(ai_settings,stats,sb,screen,ship,aliens,bullets):
    check_fleet_edges(ai_settings,aliens)
    check_aliens_bottom(ai_settings,stats,sb,screen,ship,aliens,bullets)
    aliens.update()
    #检测飞船和外星人的碰撞
    if pygame.sprite.spritecollideany(ship,aliens):
        ship_hit(ai_settings,stats,sb,screen,ship,aliens,bullets)

def ship_hit(ai_settings,stats,sb,screen,ship,aliens,bullets):
    """响应被外星人撞到的飞船"""
    if stats.ship_left > 0:
    #飞船数减一
        stats.ship_left -= 1
    #更新记分牌
        sb.prep_ships()
    #清空外星人和子弹
        aliens.empty()
        bullets.empty()
    #创建新的外星人与飞船
        create_fleet(ai_settings,screen,ship,aliens)
        ship.center_ship()
    #暂停
        sleep(0.5)
    else:
        stats.game_active=False
        pygame.mouse.set_visible(True)

def check_high_score(stats,sb):
    """检查是否产生了最高分"""
    if stats.score>stats.high_score:
        stats.high_score=stats.score
        sb.prep_high_score()
