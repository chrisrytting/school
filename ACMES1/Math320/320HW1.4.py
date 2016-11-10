def exteucalg(a, b):
    x, y, prevx, prevy = 0, 1, 1, 0
    iteration = 0
    while a:
        iteration += 1
        print 'Iteration #{}'.format(iteration)
        print 'a =', a
        q, r = b//a, b%a
        print 'q =', q
        print 'r =', r
        m, n = x-prevx*q, y-prevy*q
        print 'm =', m
        print 'n =', n
        b,a,x,y, prevx,prevy = a,r,prevx,prevy,m,n
        print 'b =', b
        print 'a =', a
        print 'x =', x
        print 'y =', y
        print 'prevx =', prevx
        print 'prevy =', prevy
    d = b
    return d, x, y

d, x, y = exteucalg(10,15)
print 'd =', d,'x =', x,'y = ',y
