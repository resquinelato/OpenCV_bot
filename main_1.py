import cv2 as cv
import numpy as np

"""Imagem Palheiro Palheiro e Agulha respectivamente
   atri1: caminho_da_imagem
   atri2: sinalizador de pré processamento da imagem None = IMREAD_UNCHANGED"""
haystack_img = cv.imread('albion_farm.jpg', cv.IMREAD_REDUCED_COLOR_2)
needle_img = cv.imread('albion_cabbage.jpg', cv.IMREAD_REDUCED_COLOR_2)

result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)

"""atri1: nome da janela
   atri2: imagem que deseja mostrar. Image Data.
cv.imshow('Result', result)
cv.waitKey()
cv.imwrite('result.jpg', haystack_img)"""

''' get the best mach position
    A localização máxima conterá a posição do pixel do canto superior esquerdo 
    para a área que mais se aproxima da imagem da agulha. O valor máximo dá
    uma indicação de quão semelhante esse achado é ao ponteiro original, onde 1 é
    perfeito e -1 é exatamente o oposto.'''
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
print('Best match top left position: %s' % str(max_loc))
print('Best match confidence: %s' % max_val)

''' If the best match value is greater than 0.8, we'll trust that we found a match'''
threshold = 0.8
if max_val >= threshold:
    print('Found needle.')

    '''Pega as dimensões da agulha.It returns a tuple of the number of rows,
        columns, and channels (if the image is color)'''
    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]
    top_left = max_loc
    bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)

    ''' Draw a rectangle on our screenshot to highlight where we found the needle.
        The line color can be set as an RGB tuple'''
    cv.rectangle(haystack_img, top_left, bottom_right, 
                    color=(0, 255, 0), thickness=2, lineType=cv.LINE_4)

    '''Ver a screenshot do processo/Salvar a imagem
        cv.imshow('Result', haystack_img)
        cv.waitKey()'''
    cv.imwrite('result.jpg', haystack_img)

else:
    print('Needle not found')