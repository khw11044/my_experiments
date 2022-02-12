from pylab import *
from numpy import *
from scipy import linalg

'''
원서 : https://books.google.co.kr/books?id=fAyODwAAQBAJ&pg=PA110&lpg=PA110&dq=python+matrix+for+DLT+solution&source=bl&ots=V2TE5uGY4t&sig=ACfU3U18ig_wbS-jNNGoFTylZL2N5JfqGw&hl=ko&sa=X&ved=2ahUKEwjs4LPA37X1AhUWs1YBHcWLDA0Q6AF6BAgVEAM
원본 : https://github.com/marktao99/python/blob/master/CVP/samples/sfm.py
변형 : https://github.com/eric-yyjau/pytorch-deepFEPE/blob/master/deepFEPE/dsac_tools/utils_F.py
Essential and Fundamental Matrices : http://www.cse.psu.edu/~rtc12/CSE486/lecture19.pdf
참고 : https://sourishghosh.com/2016/fundamental-matrix-from-camera-matrices/
참고1 : https://dellaert.github.io/19F-4476/proj3.html
참고2 : http://www.gisdeveloper.co.kr/?p=6922
'''



def compute_P(x,X):
    """    Compute camera matrix from pairs of
        2D-3D correspondences (in homog. coordinates). 
    Direct Linear Transform (DLT)
    https://www.youtube.com/watch?v=3NcQbZu6xt8
    """
    x = x.T
    X = X.T
    n = x.shape[1]
    if X.shape[1] != n:
        raise ValueError("Number of points don't match.")
        
    # create matrix for DLT solution
    M = zeros((3*n,12+n))
    for i in range(n):
        M[3*i,0:4] = X[:,i]
        M[3*i+1,4:8] = X[:,i]
        M[3*i+2,8:12] = X[:,i]
        M[3*i:3*i+3,i+12] = -x[:,i]
        
    U,S,V = linalg.svd(M)
    
    return V[-1,:12].reshape((3,4))


def triangulate_point(x1,x2,P1,P2):
    """ Point pair triangulation from 
        least squares solution. """
        
    M = zeros((6,6))
    M[:3,:4] = P1
    M[3:,:4] = P2
    M[:3,4] = -x1
    M[3:,5] = -x2

    U,S,V = linalg.svd(M)
    X = V[-1,:4]

    return X / X[3]


def triangulate(x1,x2,P1,P2):
    """    Two-view triangulation of points in 
        x1,x2 (3*n homog. coordinates). """
        
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    X = [ triangulate_point(x1[:,i],x2[:,i],P1,P2) for i in range(n)]
    return array(X).T

def calc_F(uvMat):
    A = np.zeros((len(uvMat),9))
    # img1 x' y' x y im2
    for i in range(len(uvMat)):
        A[i][0] = uvMat[i][0]*uvMat[i][2]
        A[i][1] = uvMat[i][1]*uvMat[i][2]
        A[i][2] = uvMat[i][2]
        A[i][3] = uvMat[i][0]*uvMat[i][3]
        A[i][4] = uvMat[i][1]*uvMat[i][3]
        A[i][5] = uvMat[i][3] 
        A[i][6] = uvMat[i][0]
        A[i][7] = uvMat[i][1]
        A[i][8] = 1.0  
    
    _,_,v = np.linalg.svd(A)
    # print("v", v)
    f_vec = v.transpose()[:,8]
    # print("f_vec = ", f_vec)
    f_hat = np.reshape(f_vec, (3,3))
    # print("Fmat = ", f_hat)

    # Enforce rank(F) = 2 
    s,v,d = np.linalg.svd(f_hat)
    f_hat = s @ np.diag([*v[:2], 0]) @ d

    return f_hat

def compute_fundamental(x1,x2):
    """    Computes the fundamental matrix from corresponding points 
        (x1,x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """
    
    x1 = x1.T
    x2 = x2.T
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")
    
    # build matrix for equations
    A = zeros((n,9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
            
    # compute linear least square solution
    U,S,V = linalg.svd(A)
    F = V[-1].reshape(3,3)
        
    # constrain F
    # make rank 2 by zeroing out last singular value
    U,S,V = linalg.svd(F)
    S[2] = 0
    F = dot(U,dot(diag(S),V))
    
    return F/F[2,2]

def compute_essentialmental(x1,x2):
    """    Computes the fundamental matrix from corresponding points 
        (x1,x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """
    
    x1 = x1.T
    x2 = x2.T
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")
    
    # build matrix for equations
    A = zeros((n,9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
            
    # compute linear least square solution
    U,S,V = linalg.svd(A)
    E = V[-1].reshape(3,3)
        
    # constrain F
    # make rank 2 by zeroing out last singular value
    U,S,V = linalg.svd(E)
    S[2] = 0
    E = dot(U,dot(diag(S),V))
    
    return E/E[2,2]

def compute_epipole(F):
    """ Computes the (right) epipole from a 
        fundamental matrix F. 
        (Use with F.T for left epipole.) """
    
    # return null space of F (Fx=0)
    U,S,V = linalg.svd(F)
    e = V[-1]
    return e/e[2]
    
def epipolar_line_using_F(im,F,x,epipole=None,show_epipole=True):
    """ Plot the epipole and epipolar line F*x=0
        in an image. F is the fundamental matrix 
        and x a point in the other image."""
    
    m,n = im.shape[:2]
    a,b,c = np.dot(F,x) # F @ homog_left_2d_pose
    
    # epipolar line parameter and values
    sample = np.linspace(0,n-1,10) # start, end, num
    epipolarline = np.array([(c+a*tt)/(-b) for tt in sample])

    # take only line points inside the image
    ndx = (epipolarline>=0) & (epipolarline<m) 

    return sample, epipolarline, ndx
    

def plot_epipolar_line(im,F,x,epipole=None,show_epipole=True):
    """ Plot the epipole and epipolar line F*x=0
        in an image. F is the fundamental matrix 
        and x a point in the other image."""
    
    m,n = im.shape[:2]
    line = dot(F,x)
    
    # epipolar line parameter and values
    t = linspace(0,n,100)
    lt = array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])

    # take only line points inside the image
    ndx = (lt>=0) & (lt<m) 
    plt.plot(t[ndx],lt[ndx],linewidth=2)
    
    if show_epipole:
        if epipole is None:
            epipole = compute_epipole(F)
        plt.plot(epipole[0]/epipole[2],epipole[1]/epipole[2],'r*')
    

def skew(a):
    """ Skew matrix A such that a x v = Av for any v. """

    return array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])


def compute_P_from_fundamental(F):
    """    Computes the second camera matrix (assuming P1 = [I 0]) 
        from a fundamental matrix. """
        
    e = compute_epipole(F.T) # left epipole
    Te = skew(e)
    return vstack((dot(Te,F.T).T,e)).T


def compute_P_from_essential(E):
    """    Computes the second camera matrix (assuming P1 = [I 0]) # K[I|0]이 아님
        from an essential matrix. Output is a list of four 
        possible camera matrices. """
    
    # make sure E is rank 2
    U,S,V = svd(E)
    if det(dot(U,V))<0:
        V = -V
    E = dot(U,dot(diag([1,1,0]),V))    
    
    # create matrices (Hartley p 258)
    Z = skew([0,0,-1])
    W = array([[0,-1,0],[1,0,0],[0,0,1]])
    
    # return all four solutions
    P2 = [vstack((dot(U,dot(W,V)).T,U[:,2])).T, # UWV
             vstack((dot(U,dot(W,V)).T,-U[:,2])).T,
            vstack((dot(U,dot(W.T,V)).T,U[:,2])).T,
            vstack((dot(U,dot(W.T,V)).T,-U[:,2])).T]

    return P2

def compute_rotation_translate_from_essential(E):
    '''
    참고 : https://www.youtube.com/watch?v=zX5NeY-GTO0&t=3270s
    '''
    pass


def compute_fundamental_normalized(x1,x2):
    """    Computes the fundamental matrix from corresponding points 
        (x1,x2 3*n arrays) using the normalized 8 point algorithm. """

    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = mean(x1[:2],axis=1)
    S1 = sqrt(2) / std(x1[:2])
    T1 = array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    x1 = dot(T1,x1)
    
    x2 = x2 / x2[2]
    mean_2 = mean(x2[:2],axis=1)
    S2 = sqrt(2) / std(x2[:2])
    T2 = array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2 = dot(T2,x2)

    # compute F with the normalized coordinates
    F = compute_fundamental(x1,x2)

    # reverse normalization
    F = dot(T1.T,dot(F,T2))

    return F/F[2,2]


class RansacModel(object):
    """ Class for fundmental matrix fit with ransac.py from
        http://www.scipy.org/Cookbook/RANSAC"""
    
    def __init__(self,debug=False):
        self.debug = debug
    
    def fit(self,data):
        """ Estimate fundamental matrix using eight 
            selected correspondences. """
        
        # transpose and split data into the two point sets
        data = data.T
        x1 = data[:3,:8]
        x2 = data[3:,:8]
        
        # estimate fundamental matrix and return
        F = compute_fundamental_normalized(x1,x2)
        return F
    
    def get_error(self,data,F):
        """ Compute x^T F x for all correspondences, 
            return error for each transformed point. """
        
        # transpose and split data into the two point
        data = data.T
        x1 = data[:3]
        x2 = data[3:]
        
        # Sampson distance as error measure
        Fx1 = dot(F,x1)
        Fx2 = dot(F,x2)
        denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
        err = ( diag(dot(x1.T,dot(F,x2))) )**2 / denom 
        
        # return error per point
        return err


def F_from_ransac(x1,x2,model,maxiter=5000,match_theshold=1e-6):
    """ Robust estimation of a fundamental matrix F from point 
        correspondences using RANSAC (ransac.py from
        http://www.scipy.org/Cookbook/RANSAC).
        input: x1,x2 (3*n arrays) points in hom. coordinates. """

    import ransac

    data = vstack((x1,x2))
    
    # compute F and return with inlier index
    F,ransac_data = ransac.ransac(data.T,model,8,maxiter,match_theshold,20,return_all=True)
    return F, ransac_data['inliers']