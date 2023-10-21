# warning : you have to press key while mouse is on pygame window. 

import pygame

def init():
    pygame.init()
    win = pygame.display.set_mode((400,400))

# input : keyname(ex : "a") \ output : whether or not such key is pressed. 
def getKey(keyName):
    ans = False
    for eve in pygame.event.get() : pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame,'K_{}'.format(keyName))
    if keyInput[myKey]:
        ans = True
    pygame.display.update()
    return ans

'''
def main():
    if getKey("LEFT"):
        print("left key pressed")
    elif getKey("RIGHT"):
        print("right key pressed")
'''

'''
if __name__ == '__main__':
    init()
    while True:
        main()
'''