while True:
    P = float(input('Precision:'))
    R = float(input('Recall:'))
    f1 = 2*P*R/(P+R)
    print('F1 Score: ', f1, '\n')
