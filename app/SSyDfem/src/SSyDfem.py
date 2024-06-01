# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 17:53:30 2022

@author: nf_project
"""

import numpy as np
from scipy import linalg
from scipy.sparse import bsr_matrix

def Rotation_RShell(cxyz, ref_p):
    
    # v12 = np.cross(cxyz[1,],cxyz[0,])
    # v13 = np.cross(cxyz[2,],cxyz[0,])
    v12 = cxyz[1,]-cxyz[0,]
    v13 = cxyz[2,]-cxyz[0,]
    
    vze = np.cross(v12, v13) / np.linalg.norm(np.cross(v12, v13) )
    
    if np.linalg.norm(ref_p - cxyz[0,:]) < np.linalg.norm(ref_p - cxyz[0,:] + vze):
        vze = np.cross(v13, v12) / np.linalg.norm(np.cross(v13, v12) )
    
    # % XZ plane intesection with element surface
    vxe = np.zeros(3)
    
    vxe[1] =  0 
    vxe[0] = 1/np.sqrt(1+(vze[0]/vze[2])**2)
    # vxe[0] = np.sqrt(1-vxe[0]**2)
    vxe[2] = -1/np.sqrt(1+(vze[2]/vze[0])**2)
    
      
    dd = vxe[0]*vze[0] + vxe[2]*vze[2]
    if (abs(dd) > 1e-8) :
        vxe[2] = -vxe[2]
    
    if ( (vze[2] == 0) and (vze[0] == 0)):
        vxe[0] =  1
        vxe[1] =  0
        vxe[2] =  0
    
    vye = np.cross(vze, vxe) #/ np.linalg.norm(np.cross(vze, vxe) )
    
    if (vye[1] < 0 ):
        vye=-vye
    
    Te =np.matrix([ [vxe[0] , vxe[1], vxe[2], 0,0,0    ],
                    [vye[0] , vye[1], vye[2], 0,0,0    ],
                    [vze[0] , vze[1], vze[2], 0,0,0    ],
                    [0,0,0                  , -vye[0],-vye[1],-vye[2]  ],
                    [0,0,0                  , vxe[0] , vxe[1], vxe[2]  ]#,
                    # [0,0,0                  , vye[0], vye[1],vye[2]  ],
                    # [0,0,0                  , vze[0] , vze[1], vze[2]  ]
                    ])
    # print(vze)
    return Te, vze

def Rotation_RShell2(cxyz, ref_p):
    
    # v12 = np.cross(cxyz[1,],cxyz[0,])
    # v13 = np.cross(cxyz[2,],cxyz[0,])
    v12 = cxyz[1,]-cxyz[0,]
    v13 = cxyz[2,]-cxyz[0,]
    
    vze = np.cross(v12, v13) / np.linalg.norm(np.cross(v12, v13) )
    
    if np.linalg.norm(ref_p - cxyz[0,:]) < np.linalg.norm(ref_p - cxyz[0,:] + vze):
        vze = np.cross(v13, v12) / np.linalg.norm(np.cross(v13, v12) )
    
    # % XZ plane intesection with element surface
    vxe = np.zeros(3)
    
    vxe[1] =  0 
    vxe[2] = 1/np.sqrt(1+(vze[2]/vze[0])**2)
    vxe[0] = np.sqrt(1-vxe[0]**2)
    
      
    dd = vxe[0]*vze[0] + vxe[2]*vze[2]
    if (abs(dd) > 1e-8) :
        vxe[2] = -vxe[2]
    
    if ( (vze[2] == 0) and (vze[0] == 0)):
        vxe[0] =  1
        vxe[1] =  0
        vxe[2] =  0
    
    vye = np.cross(vze, vxe) #/ np.linalg.norm(np.cross(vze, vxe) )
    
    if (vye[1] < 0 ):
        vye=-vye
    
    Te =np.matrix([ [vxe[0] , vxe[1], vxe[2] ],
                    [vye[0] , vye[1], vye[2] ],
                    [vze[0] , vze[1], vze[2] ]
                    ])
    # print(vze)
    return Te, vze

def Normal_surf(coord, elems, nE,ref_p):
    normal_e = np.ones((nE,3))   
    for i in range(nE):
        cxyz = coord[ elems[i,],:]#      % Element coordinates  
        normal_e[i,] =  np.cross(cxyz[1,]-cxyz[0,] , cxyz[2,]-cxyz[0,]) / np.linalg.norm( np.cross(cxyz[1,]-cxyz[0,] , cxyz[2,]-cxyz[0,])  )
        if np.linalg.norm(ref_p - cxyz[0,:]) < np.linalg.norm(ref_p - cxyz[0,:] + normal_e[i,]):
            normal_e[i,] =  np.cross(cxyz[2,]-cxyz[0,] , cxyz[1,]-cxyz[0,]) / np.linalg.norm( np.cross(cxyz[2,]-cxyz[0,] , cxyz[1,]-cxyz[0,])  )
        
    return normal_e

def RMat_Euler(nE,nne,gl,coord,elem):
    Le  = np.zeros(nE)
    ve  = np.zeros((nE,2))
    c1  = np.zeros(nE)
    c2  = np.zeros(nE)
    
    R_mat  = np.zeros((nE,nne*gl,nne*gl))
    
    for i in range(nE):
        Le[i]   = np.linalg.norm(coord[elem[i,0],:]-coord[elem[i,1],:])
        ve[i,:] = (coord[elem[i,1],:]-coord[elem[i,0],:])/Le[i]
        c1[i]   = ve[i,0]
        c2[i]   = ve[i,1]
        R_mat[i,:,:] = np.matrix([ [ c1[i]  , c2[i] , 0 ,  0    , 0     , 0 ] , 
                                  [ -c2[i] , c1[i] , 0 ,  0    , 0     , 0 ] ,
                                  [ 0      , 0     , 1 ,  0    , 0     , 0 ] ,
                                  [0       , 0     , 0 , c1[i] , c2[i] , 0] ,
                                  [0       , 0     , 0 ,-c2[i] , c1[i] , 0] ,
                                  [ 0      , 0     , 0 ,  0    , 0     , 1 ] 
                                ] )
    
    return R_mat, Le

def Euler2D(coord,elem, glL,glG, nE, nne, Ae, Ie, Ee):
    k_elem = np.zeros((nE,nne*glL,nne*glL))
    R_mat  = np.zeros((nE,nne*glL,nne*glG))
    K_mat  = np.zeros((nE,nne*glG,nne*glG))

    Le  = np.zeros(nE)
    ve  = np.zeros((nE,2))
    c1  = np.zeros(nE)
    c2  = np.zeros(nE)

    for i in range(nE):
        Le[i]   = np.linalg.norm(coord[elem[i,0],:]-coord[elem[i,1],:])
        ve[i,:] = (coord[elem[i,1],:]-coord[elem[i,0],:])/Le[i]
        c1[i]   = ve[i,0]
        c2[i]   = ve[i,1]

        k_elem[i,:,:] = np.matrix( [
            [Ae[i]*Ee[i]/Le[i], 0, 0, -Ae[i]*Ee[i]/Le[i], 0, 0],
            [  0.,    12*Ee[i]*Ie[i]/Le[i]**3, 6*Ee[i]*Ie[i]/Le[i]**2,    0., -12*Ee[i]*Ie[i]/Le[i]**3, 6*Ee[i]*Ie[i]/Le[i]**2 ],
            [  0.,    6*Ee[i]*Ie[i]/Le[i]**2,  4*Ee[i]*Ie[i]/Le[i],      0., -6*Ee[i]*Ie[i]/Le[i]**2,  2*Ee[i]*Ie[i]/Le[i]   ],
            [-Ae[i]*Ee[i]/Le[i], 0, 0, Ae[i]*Ee[i]/Le[i], 0, 0],
            [  0.,   -12*Ee[i]*Ie[i]/Le[i]**3,-6*Ee[i]*Ie[i]/Le[i]**2,    0.,  12*Ee[i]*Ie[i]/Le[i]**3,-6*Ee[i]*Ie[i]/Le[i]**2 ],
            [  0.,    6*Ee[i]*Ie[i]/Le[i]**2,  2*Ee[i]*Ie[i]/Le[i],      0.,  -6*Ee[i]*Ie[i]/Le[i]**2, 4*Ee[i]*Ie[i]/Le[i]   ]
            ]
            )
        
        R_mat[i,:,:] = np.matrix([ [ c1[i]  , c2[i] , 0 ,  0    , 0     , 0 ] , 
                                  [ -c2[i] , c1[i] , 0 ,  0    , 0     , 0 ] ,
                                  [ 0      , 0     , 1 ,  0    , 0     , 0 ] ,
                                  [0       , 0     , 0 , c1[i] , c2[i] , 0] ,
                                  [0       , 0     , 0 ,-c2[i] , c1[i] , 0] ,
                                  [ 0      , 0     , 0 ,  0    , 0     , 1 ] 
                                ] )
        
        K_mat[i,:,:] = np.dot( np.dot(  np.transpose(R_mat[i,:,:]) , k_elem[i,:,:]) , R_mat[i,:,:])
        # print(ve)
    
    return K_mat, R_mat, Le

def Euler2D_dyn(coord,elem, glL,glG, nE, nne, Ae, Ie, Ee,Rhoe, option_e):
    k_elem = np.zeros((nE,nne*glL,nne*glL))
    m_elem = np.zeros((nE,nne*glL,nne*glL))
    R_mat  = np.zeros((nE,nne*glL,nne*glG))
    K_mat  = np.zeros((nE,nne*glG,nne*glG))
    M_mat  = np.zeros((nE,nne*glG,nne*glG))


    Le  = np.zeros(nE)
    ve  = np.zeros((nE,2))
    c1  = np.zeros(nE)
    c2  = np.zeros(nE)

    for i in range(nE):
        Le[i]   = np.linalg.norm(coord[elem[i,0],:]-coord[elem[i,1],:])
        ve[i,:] = (coord[elem[i,1],:]-coord[elem[i,0],:])/Le[i]
        c1[i]   = ve[i,0]
        c2[i]   = ve[i,1]

        k_elem[i,:,:] = np.matrix( [
            [Ae[i]*Ee[i]/Le[i], 0, 0, -Ae[i]*Ee[i]/Le[i], 0, 0],
            [  0.,    12*Ee[i]*Ie[i]/Le[i]**3, 6*Ee[i]*Ie[i]/Le[i]**2,    0., -12*Ee[i]*Ie[i]/Le[i]**3, 6*Ee[i]*Ie[i]/Le[i]**2 ],
            [  0.,    6*Ee[i]*Ie[i]/Le[i]**2,  4*Ee[i]*Ie[i]/Le[i],      0., -6*Ee[i]*Ie[i]/Le[i]**2,  2*Ee[i]*Ie[i]/Le[i]   ],
            [-Ae[i]*Ee[i]/Le[i], 0, 0, Ae[i]*Ee[i]/Le[i], 0, 0],
            [  0.,   -12*Ee[i]*Ie[i]/Le[i]**3,-6*Ee[i]*Ie[i]/Le[i]**2,    0.,  12*Ee[i]*Ie[i]/Le[i]**3,-6*Ee[i]*Ie[i]/Le[i]**2 ],
            [  0.,    6*Ee[i]*Ie[i]/Le[i]**2,  2*Ee[i]*Ie[i]/Le[i],      0.,  -6*Ee[i]*Ie[i]/Le[i]**2, 4*Ee[i]*Ie[i]/Le[i]   ]
            ]
            )
        
        R_mat[i,:,:] = np.matrix([ [ c1[i]  , c2[i] , 0 ,  0    , 0     , 0 ] , 
                                  [ -c2[i] , c1[i] , 0 ,  0    , 0     , 0 ] ,
                                  [ 0      , 0     , 1 ,  0    , 0     , 0 ] ,
                                  [0       , 0     , 0 , c1[i] , c2[i] , 0] ,
                                  [0       , 0     , 0 ,-c2[i] , c1[i] , 0] ,
                                  [ 0      , 0     , 0 ,  0    , 0     , 1 ] 
                                ] )
        
        K_mat[i,:,:] = np.dot( np.dot(  np.transpose(R_mat[i,:,:]) , k_elem[i,:,:]) , R_mat[i,:,:])
        # print(ve)
        
        
        if option_e[i] == 0:
            m_elem[i,:,:] = Rhoe[i] * Ae[i] * Le[i] * np.matrix( [
                [1/3 , 0             , 0              , 1/6 , 0            , 0             ],
                [ 0  , 13/35         , 11*Le[i] /210  , 0   , 9/70         , -13*Le[i]/420  ],
                [ 0  , 11*Le[i] /210, Le[i]**2/105   , 0   , 13*Le[i]/420 , -Le[i]**2/140  ],
                [ 1/6, 0             , 0              , 1/3 , 0            , 0             ],
                [ 0  , 9/70          , 13*Le[i]/420   , 0   , 13/35        , -11*Le[i]/210 ],
                [ 0  , -13*Le[i]/420 , -Le[i]**2/140  , 0   , -11*Le[i]/210, Le[i]**2/105  ]
                ] )
            
        elif option_e[i] == 1:
            m_elem[i,:,:] = Rhoe[i]/2 * Ae[i] * Le[i] * np.matrix( [
                [1 ,0 ,0 ,0 ,0 ,0],
                [0 ,1 ,0 ,0 ,0 ,0],
                [0 ,0 , Le[i]**2/12 ,0 ,0 ,0],
                [0 ,0 ,0 ,1 ,0 ,0],
                [0 ,0 ,0 ,0 ,1 ,0],
                [0 ,0 ,0 ,0 ,0 ,Le[i]**2/12]
                ])
        M_mat[i,:,:] = np.dot( np.dot(  np.transpose(R_mat[i,:,:]) , m_elem[i,:,:]) , R_mat[i,:,:])
        # elif option_e[i] == 2:
        #     m_elem[i,:,:] = Rhoe[i] * Ae[i] * Le[i] * np.matrix( [
                
        #         ])
    
    return K_mat, M_mat, R_mat, Le

def Euler3D(coord,elem, glL,glG, nE, nne, Ae, Iye, Ize, Je,  Ee, Ge, orient_e):
    k_elem = np.zeros((nE,nne*glL,nne*glL))
    R_mat  = np.zeros((nE,nne*glL,nne*glG))
    K_mat  = np.zeros((nE,nne*glG,nne*glG))

    Le  = np.zeros(nE)
    ve  = np.zeros((nE,3))
    L13e  = np.zeros(nE)
    v13e  = np.zeros((nE,3))
    

    l = np.zeros(3)    
    m = np.zeros(3)
    n = np.zeros(3)
    
    
    
    for i in range(nE):
        Le[i]   = np.linalg.norm(coord[elem[i,0],:]-coord[elem[i,1],:])
        ve[i,:] = (coord[elem[i,1],:]-coord[elem[i,0],:])/Le[i]
        p3 = (coord[elem[i,1],:]-coord[elem[i,0],:])/2 + coord[elem[i,0],:] +orient_e[i]
        L13e[i]   = np.linalg.norm(p3- coord[elem[i,1],:])
        v13e[i,:] = ( p3- coord[elem[i,0],:] ) / L13e[i]
        
        a = Ee[i]*Ae[i]/Le[i]
        b = 12*Ee[i]*Ize[i]/Le[i]**3
        c = 6*Ee[i]*Ize[i]/Le[i]**2
        d = 12*Ee[i]*Iye[i]/Le[i]**3
        e = 6*Ee[i]*Iye[i]/Le[i]**2
        f = Ge[i]*Je[i]/Le[i]
        g = 2*Ee[i]*Iye[i]/Le[i]
        h = 2*Ee[i]*Ize[i]/Le[i]
        

        k_elem[i,:,:] = np.matrix( [
            [ a, 0, 0, 0, 0,   0,  -a, 0, 0, 0, 0,   0  ],
            [ 0, b, 0, 0, 0,   c,   0,-b, 0, 0, 0,   c  ],
            [ 0, 0, d, 0,-e,   0,   0, 0,-d, 0,-e,   0  ],
            [ 0, 0, 0, f, 0,   0,   0, 0, 0,-f, 0,   0  ],
            [ 0, 0,-e, 0, 2*g, 0,   0, 0, e, 0, g,   0  ],
            [ 0, c, 0, 0, 0,   2*h, 0,-c, 0, 0, 0,   h  ],
            [-a, 0, 0, 0, 0,   0,   a, 0, 0, 0, 0,   0  ],
            [ 0,-b, 0, 0, 0,  -c,   0, b, 0, 0, 0,  -c  ],
            [ 0, 0,-d, 0, e,   0,   0, 0, d, 0, e,   0  ],
            [ 0, 0, 0,-f, 0,   0,   0, 0, 0, f, 0,   0  ],
            [ 0, 0,-e, 0, g,   0,   0, 0, e, 0, 2*g, 0  ],
            [ 0, c, 0, 0, 0,   h,   0,-c, 0, 0, 0,   2*h]
            ]
            )
        # print( k_elem[i,:,:])
        
        l[0] = ve[i,0]
        m[0] = ve[i,1]
        n[0] = ve[i,2]
        
        Vz = np.cross(ve[i,:],v13e[i,:]) / np.linalg.norm (np.cross(ve[i,:],v13e[i,:]))
        
        
        l[2] = Vz[0]
        m[2] = Vz[1]
        n[2] = Vz[2]
        
        Vy = np.cross(Vz,ve[i,:]) 
    
        
        l[1] = Vy[0]
        m[1] = Vy[1]
        n[1] = Vy[2]
        
        lambd = np.matrix([ [l[0],m[0],n[0]],
                            [l[1],m[1],n[1]],
                            [l[2],m[2],n[2]]
                            ])
    
        R_mat[i,:,:] = np.matrix( np.zeros(glL*nne*glG*nne).reshape(12,12))
        
        R_mat[i,0:3,0:3] =lambd
        R_mat[i,3:6,3:6] =lambd
        R_mat[i,6:9,6:9] =lambd
        R_mat[i,9:,9:] =lambd
                
        K_mat[i,:,:] = np.dot( np.dot(  np.transpose(R_mat[i,:,:]) , k_elem[i,:,:]) , R_mat[i,:,:])
    
    return K_mat, R_mat, Le


def Euler3D_dyn(coord,elem, glL,glG, nE, nne, Ae, Iye, Ize, Je,  Ee, Ge, orient_e, Rhoe, option_e):
    k_elem = np.zeros((nE,nne*glL,nne*glL))
    m_elem = np.zeros((nE,nne*glL,nne*glL))
    R_mat  = np.zeros((nE,nne*glL,nne*glG))
    K_mat  = np.zeros((nE,nne*glG,nne*glG))
    M_mat  = np.zeros((nE,nne*glG,nne*glG))

    Le  = np.zeros(nE)
    ve  = np.zeros((nE,3))
    L13e  = np.zeros(nE)
    v13e  = np.zeros((nE,3))
    

    l = np.zeros(3)    
    m = np.zeros(3)
    n = np.zeros(3)
    
    
    
    for i in range(nE):
        Le[i]   = np.linalg.norm(coord[elem[i,0],:]-coord[elem[i,1],:])
        ve[i,:] = (coord[elem[i,1],:]-coord[elem[i,0],:])/Le[i]
        p3 = (coord[elem[i,1],:]-coord[elem[i,0],:])/2 + coord[elem[i,0],:] +orient_e[i]
        L13e[i]   = np.linalg.norm(p3- coord[elem[i,1],:])
        v13e[i,:] = ( p3- coord[elem[i,0],:] ) / L13e[i]
        
        a = Ee[i]*Ae[i]/Le[i]
        b = 12*Ee[i]*Ize[i]/Le[i]**3
        c = 6*Ee[i]*Ize[i]/Le[i]**2
        d = 12*Ee[i]*Iye[i]/Le[i]**3
        e = 6*Ee[i]*Iye[i]/Le[i]**2
        f = Ge[i]*Je[i]/Le[i]
        g = 2*Ee[i]*Iye[i]/Le[i]
        h = 2*Ee[i]*Ize[i]/Le[i]
        

        k_elem[i,:,:] = np.matrix( [
            [ a, 0, 0, 0, 0,   0,  -a, 0, 0, 0, 0,   0  ],
            [ 0, b, 0, 0, 0,   c,   0,-b, 0, 0, 0,   c  ],
            [ 0, 0, d, 0,-e,   0,   0, 0,-d, 0,-e,   0  ],
            [ 0, 0, 0, f, 0,   0,   0, 0, 0,-f, 0,   0  ],
            [ 0, 0,-e, 0, 2*g, 0,   0, 0, e, 0, g,   0  ],
            [ 0, c, 0, 0, 0,   2*h, 0,-c, 0, 0, 0,   h  ],
            [-a, 0, 0, 0, 0,   0,   a, 0, 0, 0, 0,   0  ],
            [ 0,-b, 0, 0, 0,  -c,   0, b, 0, 0, 0,  -c  ],
            [ 0, 0,-d, 0, e,   0,   0, 0, d, 0, e,   0  ],
            [ 0, 0, 0,-f, 0,   0,   0, 0, 0, f, 0,   0  ],
            [ 0, 0,-e, 0, g,   0,   0, 0, e, 0, 2*g, 0  ],
            [ 0, c, 0, 0, 0,   h,   0,-c, 0, 0, 0,   2*h]
            ]
            )
        # print( k_elem[i,:,:])
        
        l[0] = ve[i,0]
        m[0] = ve[i,1]
        n[0] = ve[i,2]
        
        Vz = np.cross(ve[i,:],v13e[i,:]) / np.linalg.norm (np.cross(ve[i,:],v13e[i,:]))
        
        
        l[2] = Vz[0]
        m[2] = Vz[1]
        n[2] = Vz[2]
        
        Vy = np.cross(Vz,ve[i,:]) 
    
        
        l[1] = Vy[0]
        m[1] = Vy[1]
        n[1] = Vy[2]
        
        lambd = np.matrix([ [l[0],m[0],n[0]],
                            [l[1],m[1],n[1]],
                            [l[2],m[2],n[2]]
                            ])
    
        R_mat[i,:,:] = np.matrix( np.zeros(glL*nne*glG*nne).reshape(12,12))
        
        R_mat[i,0:3,0:3] =lambd
        R_mat[i,3:6,3:6] =lambd
        R_mat[i,6:9,6:9] =lambd
        R_mat[i,9:,9:] =lambd
                
        K_mat[i,:,:] = np.dot( np.dot(  np.transpose(R_mat[i,:,:]) , k_elem[i,:,:]) , R_mat[i,:,:])
        
        if option_e[i] == 0:
            m_elem[i,:,:] = Rhoe[i] * Ae[i] * Le[i] * np.matrix( [
                [1/3 , 0             , 0              , 0             , 0              , 0             , 1/6 , 0             , 0             , 0             , 0              , 0             ],
                [ 0  , 13/35         , 0              , 0             , 0              , 11*Le[i] /210 , 0   , 9/70          , 0             , 0             , 0              , -13*Le[i]/420  ],
                [ 0  , 0             , 13/35          , 0             , -11*Le[i] /210 , 0             , 0   , 0             , 9/70          , 0             , 13*Le[i]/420   , 0             ],
                [ 0  , 0             , 0              , Je[i]/3/Ae[i] , 0              , 0             , 0   , 0             , 0             , Je[i]/6/Ae[i] , 0              , 0             ],
                [ 0  , 0             , -11*Le[i] /210 , 0             , Le[i]**2/105   , 0             , 0   , 0             , -13*Le[i]/420  , 0             , -Le[i]**2/140  , 0             ],
                [ 0  , 11*Le[i] /210 , 0              , 0             , 0              , Le[i]**2/105  , 0   , 13*Le[i]/420 , 0             , 0             , 0              , -Le[i]**2/140 ],
                [ 1/6, 0             , 0              , 0             , 0              , 0             , 1/3 , 0             , 0             , 0             , 0              , 0             ],
                [ 0  , 9/70          , 0              , 0             , 0              , 13*Le[i]/420  , 0   , 13/35         , 0             , 0             , 0              , -11*Le[i] /210],
                [ 0  , 0             , 9/70           , 0             ,-13*Le[i]/420   , 0             , 0   , 0             , 13/35         , 0             , 11*Le[i] /210 , 0             ],
                [ 0  , 0             , 0              , Je[i]/6/Ae[i] , 0              , 0             , 0   , 0             , 0             , Je[i]/3/Ae[i] , 0             , 0             ],
                [ 0  , 0             , 13*Le[i]/420   , 0             , -Le[i]**2/140  , 0             , 0   , 0             , 11*Le[i] /210 , 0             , Le[i]**2/105  , 0             ],
                [ 0  , -13*Le[i]/420 , 0              , 0             , 0              , -Le[i]**2/140 , 0   ,-11*Le[i] /210 , 0             , 0             , 0              , Le[i]**2/105  ]
                ] )
            
        elif option_e[i] == 1:
            m_elem[i,:,:] = Rhoe[i]/2 * Ae[i] * Le[i] * np.matrix( [
                [1 ,0 ,0 ,0 ,0 ,0],
                [0 ,1 ,0 ,0 ,0 ,0],
                [0 ,0 , Le[i]**2/12 ,0 ,0 ,0],
                [0 ,0 ,0 ,1 ,0 ,0],
                [0 ,0 ,0 ,0 ,1 ,0],
                [0 ,0 ,0 ,0 ,0 ,Le[i]**2/12]
                ])
        M_mat[i,:,:] = np.dot( np.dot(  np.transpose(R_mat[i,:,:]) , m_elem[i,:,:]) , R_mat[i,:,:])
    
    return K_mat, M_mat, R_mat, Le

def Truss2D(coord,elem, glL,glG, nE, nne, Ae, Ee):
    k_elem = np.zeros((nE,nne*glL,nne*glL))
    R_mat  = np.zeros((nE,nne*glL,nne*glG))
    K_mat  = np.zeros((nE,nne*glG,nne*glG))

    Le  = np.zeros(nE)
    ve  = np.zeros((nE,2))
    c1  = np.zeros(nE)
    c2  = np.zeros(nE)

    for i in range(nE):
        Le[i]   = np.linalg.norm(coord[elem[i,0],:]-coord[elem[i,1],:])
        ve[i,:] = (coord[elem[i,1],:]-coord[elem[i,0],:])/Le[i]
        c1[i]   = ve[i,0]
        c2[i]   = ve[i,1]

        k_elem[i,:,:] = np.matrix( [
            [Ae[i]*Ee[i]/Le[i]  , -Ae[i]*Ee[i]/Le[i] ],
            [-Ae[i]*Ee[i]/Le[i] , Ae[i]*Ee[i]/Le[i]  ]
            ]
            )
        
        R_mat[i,:,:] = np.matrix([ [ c1[i]  , c2[i] ,  0    , 0     ] , 
                                  [0       , 0      , c1[i] , c2[i] ] 
                                 ] )
        
        K_mat[i,:,:] = np.dot( np.dot(  np.transpose(R_mat[i,:,:]) , k_elem[i,:,:]) , R_mat[i,:,:])
        # print(ve)
    
    return K_mat, R_mat, Le

def Truss2D_dyn(coord,elem, glL,glG, nE, nne, Ae, Ee, Rhoe, option_e):
    k_elem = np.zeros((nE,nne*glL,nne*glL))
    m_elem = np.zeros((nE,nne*glG,nne*glG))
    R_mat  = np.zeros((nE,nne*glL,nne*glG))
    K_mat  = np.zeros((nE,nne*glG,nne*glG))
    M_mat  = np.zeros((nE,nne*glG,nne*glG))
    

    Le  = np.zeros(nE)
    ve  = np.zeros((nE,2))
    c1  = np.zeros(nE)
    c2  = np.zeros(nE)

    for i in range(nE):
        Le[i]   = np.linalg.norm(coord[elem[i,0],:]-coord[elem[i,1],:])
        ve[i,:] = (coord[elem[i,1],:]-coord[elem[i,0],:])/Le[i]
        c1[i]   = ve[i,0]
        c2[i]   = ve[i,1]

        k_elem[i,:,:] = np.matrix( [
            [Ae[i]*Ee[i]/Le[i]  , -Ae[i]*Ee[i]/Le[i] ],
            [-Ae[i]*Ee[i]/Le[i] , Ae[i]*Ee[i]/Le[i]  ]
            ]
            )
        
        m_elem[i,:,:] = Rhoe[i] * Ae[i] * Le[i] /2 * np.matrix( [
                [1 ,0 , 0 , 0],
                        [0 , 1 ,0 ,0 ],
                        [0, 0 , 1, 0],
                        [0, 0, 0, 1]
                ]
                )
        
        R_mat[i,:,:] = np.matrix([ [ c1[i]  , c2[i] ,  0    , 0     ] , 
                                  [0       , 0      , c1[i] , c2[i] ] 
                                 ] )
        
        # R_mat2 = np.matrix([ [ c1[i]  , c2[i] ,  0    , 0     ] , 
        #                     [  -c2[i]  , c1[i] ,  0    , 0     ] , 
        #                     [ 0  , 0 ,     c1[i]  , c2[i]     ] , 
        #                           [0       , 0      , -c2[i]  , c1[i] ] 
        #                          ] )
        
        K_mat[i,:,:] = np.dot( np.dot(  np.transpose(R_mat[i,:,:]) , k_elem[i,:,:]) , R_mat[i,:,:])
        
        # M_mat[i,:,:] = np.dot( np.dot(  np.transpose(R_mat2) , m_elem[i,:,:]) , R_mat2)
        # print(ve)
        
        if option_e[i]:
            M_mat[i,:,:] = Rhoe[i] * Ae[i] * Le[i] /6 * np.matrix( [
                [2 ,0 , 1 , 0],
                [0 , 2 ,0 ,1 ],
                [1, 0 , 2, 0],
                [0, 1, 0, 2]
                ]
                )
        else:
            M_mat[i,:,:] = Rhoe[i] * Ae[i] * Le[i] /6 * np.matrix( [
                [2 ,0 , 1 , 0],
                [0 , 2 ,0 ,1 ],
                [1, 0 , 2, 0],
                [0, 1, 0, 2]
                ]
                )
    
    return K_mat, M_mat,  R_mat, Le

def shell_QLLL_v2(coord,elem, glL,glG, nE, nne, Ee, poi_e, thick_e,denss_e, groups_shell_e, nndof,ref_p, copl):
    # Gauss point coordinates
    gauss_x = np.zeros(4)
    gauss_y = np.zeros(4)
    gauss_w = np.zeros(4)
    
    gauss_x[0] = -1/np.sqrt(3)
    gauss_y[0] = -1/np.sqrt(3)
    gauss_w[0] = 1
    
    gauss_x[1] =  1/np.sqrt(3)
    gauss_y[1] = -1/np.sqrt(3)
    gauss_w[1] = 1
    
    gauss_x[2] =  1/np.sqrt(3)
    gauss_y[2] =  1/np.sqrt(3)
    gauss_w[2] = 1
    
    gauss_x[3] = -1/np.sqrt(3)
    gauss_y[3] =  1/np.sqrt(3)
    gauss_w[3] = 1
    
    c_xgs = np.array( [0,1,0,-1] ) /3**0.5
    c_ygs = np.array([-1 , 0 , 1 ,  0 ]) /3**0.5
    
    # matriz de rigidez del sistema a rellenar
    K_ =  bsr_matrix((nndof,nndof), dtype=np.float64).toarray()
    
    index_1 = [0,6,12,18]
    # index_2 = [i1,i2,i3,i4]
    index_3 = [0,5,10,15]
    
    # K_ = np.zeros((nndof,nndof))
    
    ic = 0 
    
    if groups_shell_e[0] ==0:
        for i in range(nE):
            if i == 0:
                # Material properties(Constant over the domain)              
                Dp = Ee[i] / (1-poi_e[i]**2)*  np.matrix([[1       ,  poi_e[i],       0],
                                                          [poi_e[i],   1      ,       0],
                                                          [   0    ,   0      ,    0.5*(1-poi_e[i]) ] ])             
                Dm = thick_e[i] * Dp
                Db = thick_e[i]**3/12*Dp
                Ds = 5/6*thick_e[i]*Ee[i]/2/(1+poi_e[i]) * np.matrix( [[1,0],[0,1]] )
                
            cxyz = coord[ elem[i,],:]#      % Element coordinates  
            # Transformacion
            Le, normal = Rotation_RShell(cxyz[0:3,:],ref_p)
            
            # Te = bsr_matrix((nne*glG,nne*glG),dtype=np.float64()).toarray()
            Te = bsr_matrix((nne*glL,nne*glG),dtype=np.float64()).toarray()
                        
            # coordenadas locales
            ctxy = np.zeros((4,3))
            
            for l in range(4):
                ctxy[l,:] = np.dot(Le[:3,:3] ,  cxyz[l,:])#    % Rotate coordinates to element mid plane       
           
            # tole=1e-10
            # ctxy = np.where(np.abs(ctxy)<tole,0,ctxy)
            # print(cxyz)
            # print(ctxy)
            # Posiciones en la matriz de rigidez
            i1 = glG*elem[i,0]
            i2 = glG*elem[i,1]
            i3 = glG*elem[i,2]
            i4 = glG*elem[i,3]
            
            # index_1 = [0,6,12,18]
            index_2 = [i1,i2,i3,i4]
            # index_3 = [0,5,10,15]
            # print(elem[i,])
            # print(index_2)
            
            K_elem_1 = np.zeros((nne*glL,nne*glL),dtype=np.float64)
            K_elem_2 = np.zeros((nne*glG,nne*glG),dtype=np.float64)         
            # K_elem_2 = np.zeros((glG,glG),dtype=np.float64)         
            K_elem = np.zeros((nne*glG,nne*glG),dtype=np.float64)

            
            for j1 in range(4):
                Bm, Bb, Bs, Jac_module = func_B_shell(gauss_x[j1],gauss_y[j1],c_xgs[j1],c_ygs[j1],ctxy[:,0],ctxy[:,1])
                # Bm, Bb, Bs, Jac_module = func_B_shell(gauss_x[j1],gauss_y[j1],gauss_x[j1],gauss_y[j1],ctxy[:,0],ctxy[:,1])
                Km = np.transpose(Bm) @ Dm @ Bm*Jac_module
                Kb = np.transpose(Bb) @ Db @ Bb*Jac_module
                Ks = np.transpose(Bs) @ Ds @ Bs*Jac_module
                K_elem_1 = K_elem_1 +Km +Kb +Ks
                      
                # Te[index_1[j1]:index_1[j1]+glG,index_1[j1]:index_1[j1]+glG] = Le
                Te[index_3[j1]:index_3[j1]+glL,index_1[j1]:index_1[j1]+glG] = Le
            # print(np.shape(Te))
                                  
            K_elem = np.dot(np.dot (np.transpose(Te) , K_elem_1) , Te)
            
            
            if i== copl[ic][0]:
                # print(copl[ic])
                k_theta = Jac_module*Ee[i]*thick_e[i]
                    
                T_k_theta = np.matrix([ [ normal[0]**2      ,normal[0]*normal[1] , normal[0] *normal[2]] ,
                                        [ normal[1]*normal[0] ,normal[1]**2        , normal[1]*normal[2] ] ,
                                        [ normal[2]*normal[0] ,normal[2]*normal[1]   ,normal[2]**2 ] ])
                # K_elem_2[3:,3:] = k_theta * T_k_theta
                
                index = copl[ic][1:]
                index.sort()
                # print(index)
                
                for j1 in range(len(index)):
                    for j2 in range(len(index)):
                        a = glG*index[j1]                    
                        b = glG*index[j2]                        
                        K_elem_2[a+3:a+glG,b+3:b+glG] = k_theta * T_k_theta
                ic += 1
                K_elem_G  =K_elem +K_elem_2
            else:
                K_elem_G  =K_elem
        
            for j3 in range(4):
                for j4 in range(4):
                    a = index_2[j3]  
                    c = index_1[j3]

                    b = index_2[j4]
                    d = index_1[j4]
                    # print([a,b,c,d])
                    K_[a:a+glG,b:b+glG] =  K_[a:a+glG,b:b+glG] + K_elem_G[c:c+glG,d:d+glG] 
         
    return K_ *0.5 + np.transpose(K_) *0.5#,K_elem,K_elem_1

def func_B_shell(xgs,ygs,xcgs,ycgs,x,y):

    N1 = (1-xgs)*(1-ygs)/4 
    N2 = (1+xgs)*(1-ygs)/4 
    N3 = (1+xgs)*(1+ygs)/4 
    N4 = (1-xgs)*(1+ygs)/4 
    
    jac_m = np.zeros((2,2))
    jac_m = np.asmatrix(jac_m)
    
    # Derivadas de las funciones de interpolacion
    dxN1_ = (-1+ygs)/4
    dxN2_ = ( 1-ygs)/4
    dxN3_ = ( 1+ygs)/4
    dxN4_ = (-1-ygs)/4
    
    dyN1_ = (-1+xgs)/4
    dyN2_ = (-1-xgs)/4
    dyN3_ = ( 1+xgs)/4
    dyN4_ = (1-xgs)/4
    
    jac_m = np.dot ( np.matrix( [ [ dxN1_, dxN2_ , dxN3_ , dxN4_ ] ,
                                  [ dyN1_, dyN2_ , dyN3_ , dyN4_ ] 
                                  ]  ),
                     np.matrix( [ [ x[0] , y[0] ],
                                  [ x[1] , y[1] ],
                                  [ x[2] , y[2] ],
                                  [ x[3] , y[3] ]
                                  ] ) ) 
    jac_m_inv = np.linalg.inv(jac_m)
    # print(jac_m)
    Jac_module = np.abs(np.linalg.det( jac_m))
    
    dxN1 = jac_m_inv[0,0]*dxN1_ + jac_m_inv[0,1]*dyN1_
    dyN1 = jac_m_inv[1,0]*dxN1_ + jac_m_inv[1,1]*dyN1_
    
    dxN2 = jac_m_inv[0,0]*dxN2_ + jac_m_inv[0,1]*dyN2_
    dyN2 = jac_m_inv[1,0]*dxN2_ + jac_m_inv[1,1]*dyN2_
    
    dxN3 = jac_m_inv[0,0]*dxN3_ + jac_m_inv[0,1]*dyN3_
    dyN3 = jac_m_inv[1,0]*dxN3_ + jac_m_inv[1,1]*dyN3_
    
    dxN4 = jac_m_inv[0,0]*dxN4_ + jac_m_inv[0,1]*dyN4_     
    dyN4 = jac_m_inv[1,0]*dxN4_ + jac_m_inv[1,1]*dyN4_
    
    # Bm = [ Bm1 Bm2 Bm3 Bm4 ]
    Bm = np.matrix( [ 
        [dxN1,0   ,0,0,0 ,dxN2,0   ,0,0,0  ,dxN3,0   ,0,0,0 ,dxN4,0   ,0,0,0],
        [0,dyN1   ,0,0,0 ,0,dyN2   ,0,0,0  ,0,dyN3   ,0,0,0 ,0,dyN4   ,0,0,0],
        [dyN1,dxN1,0,0,0 ,dyN2,dxN2,0,0,0  ,dyN3,dxN3,0,0,0 ,dyN4,dxN4,0,0,0]
        ])
    # print("Bm")
    # print(Bm)
    # Bb = -np.matrix( [ 
    #     [0,0,0,dxN1,0,    0,0,0,dxN2,0 ,   0,0,0,dxN3,0   , 0,0,0 ,dxN4,0   ],
    #     [0,0,0,0,dyN1,    0,0,0,0,dyN2,    0,0,0,0,dyN3   , 0,0,0,0,dyN4    ],
    #     [0,0,0,dyN1,dxN1, 0,0,0,dyN2,dxN2, 0,0,0,dyN3,dxN3, 0,0,0,dyN4,dxN4 ]
    #     ])
    
    Bb = np.matrix( [ 
        [0,0,0,dxN1,0,    0,0,0,dxN2,0 ,   0,0,0,dxN3,0   , 0,0,0 ,dxN4,0    ],
        [0,0,0,0   ,dyN1, 0,0,0,0   ,dyN2, 0,0,0,0   ,dyN3, 0,0,0,0    ,dyN4 ],
        [0,0,0,dyN1,dxN1, 0,0,0,dyN2,dxN2, 0,0,0,dyN3,dxN3, 0,0,0,dyN4,dxN4  ]
        ])
    
    
    # print("Bb")
    # print(Bb)
    # Bb = np.matrix( [ 
    #     [0,0,0,0,-dxN1,    0,0,0,0,-dxN2,    0,0,0,0,-dxN3   , 0,0,0,0,-dxN4    ],
    #     [0,0,0,dyN1,0,    0,0,0,dyN2,0,    0,0,0,dyN3,0   , 0,0,0,dyN4,0    ],
    #     [0,0,0,dxN1,-dyN1, 0,0,0,dxN2,-dyN2, 0,0,0,dxN3,-dyN3, 0,0,0,dxN4,-dyN4 ]
    #     ])

    # Bb = np.matrix( [ 
    #     [0,0,0,0,dxN1,    0,0,0,0,dxN2,    0,0,0,0,dxN3   , 0,0,0,0,dxN4    ],
    #     [0,0,0,dyN1,0,    0,0,0,dyN2,0,    0,0,0,dyN3,0   , 0,0,0,dyN4,0    ],
    #     [0,0,0,dxN1,dyN1, 0,0,0,dxN2,dyN2, 0,0,0,dxN3,dyN3, 0,0,0,dxN4,dyN4 ]
    #     ])
    
    # Bs = np.matrix( [ 
    #     [0,0,dxN1, N1,0, 0,0,dxN2, N2,0, 0,0,dxN3, N3,0  ,0,0,dxN4, N4,0],
    #     [0,0,dyN1,0, N1, 0,0,dyN2,0, N2, 0,0,dyN3,0, N3  ,0,0,dyN4,0, N4]
    #     ]
    
    dxN1_ = (-1+ycgs)/4
    dxN2_ = ( 1-ycgs)/4
    dxN3_ = ( 1+ycgs)/4
    dxN4_ = (-1-ycgs)/4
    
    dyN1_ = (-1+xcgs)/4
    dyN2_ = (-1-xcgs)/4
    dyN3_ = ( 1+xcgs)/4
    dyN4_ = (1-xcgs)/4
    
    dxN1 = jac_m_inv[0,0]*dxN1_ + jac_m_inv[0,1]*dyN1_
    dyN1 = jac_m_inv[1,0]*dxN1_ + jac_m_inv[1,1]*dyN1_
    
    dxN2 = jac_m_inv[0,0]*dxN2_ + jac_m_inv[0,1]*dyN2_
    dyN2 = jac_m_inv[1,0]*dxN2_ + jac_m_inv[1,1]*dyN2_
    
    dxN3 = jac_m_inv[0,0]*dxN3_ + jac_m_inv[0,1]*dyN3_
    dyN3 = jac_m_inv[1,0]*dxN3_ + jac_m_inv[1,1]*dyN3_
    
    dxN4 = jac_m_inv[0,0]*dxN4_ + jac_m_inv[0,1]*dyN4_     
    dyN4 = jac_m_inv[1,0]*dxN4_ + jac_m_inv[1,1]*dyN4_
    
    # Bs = np.matrix( [ 
    #     [0,0,dxN1, N1,0, 0,0,dxN1, N1,0, 0,0,dxN1, N1,0  ,0,0,dxN1, N1,0],
    #     [0,0,dyN1,0, N1, 0,0,dyN1,0, N1, 0,0,dyN1,0, N1  ,0,0,dyN1,0, N1],
        
    #     [0,0,dxN2, N2,0, 0,0,dxN2, N2,0, 0,0,dxN2, N2,0  ,0,0,dxN2, N2,0],
    #     [0,0,dyN2,0, N2, 0,0,dyN2,0, N2, 0,0,dyN2,0, N2  ,0,0,dyN2,0, N2],
        
    #     [0,0,dxN3, N3,0, 0,0,dxN3, N3,0, 0,0,dxN3, N3,0  ,0,0,dxN3, N3,0],
    #     [0,0,dyN3,0, N3, 0,0,dyN3,0, N3, 0,0,dyN3,0, N3  ,0,0,dyN3,0, N3],
        
    #     [0,0,dxN4, N4,0, 0,0,dxN4, N4,0, 0,0,dxN4, N4,0  ,0,0,dxN4, N4,0],
    #     [0,0,dyN4,0, N4, 0,0,dyN4,0, N4, 0,0,dyN4,0, N4  ,0,0,dyN4,0, N4]
    #     ])
    
    Bs = np.matrix( [ 
        [0,0,dxN1, -N1,0, 0,0,dxN1, -N1,0, 0,0,dxN1, -N1,0  ,0,0,dxN1, -N1,0],
        [0,0,dyN1,0, -N1, 0,0,dyN1,0, -N1, 0,0,dyN1,0, -N1  ,0,0,dyN1,0, -N1],
        
        [0,0,dxN2, -N2,0, 0,0,dxN2, -N2,0, 0,0,dxN2, -N2,0  ,0,0,dxN2, -N2,0],
        [0,0,dyN2,0, -N2, 0,0,dyN2,0, -N2, 0,0,dyN2,0, -N2  ,0,0,dyN2,0, -N2],
        
        [0,0,dxN3, -N3,0, 0,0,dxN3, -N3,0, 0,0,dxN3, -N3,0  ,0,0,dxN3, -N3,0],
        [0,0,dyN3,0, -N3, 0,0,dyN3,0, -N3, 0,0,dyN3,0, -N3  ,0,0,dyN3,0, -N3],
        
        [0,0,dxN4, -N4,0, 0,0,dxN4, -N4,0, 0,0,dxN4, -N4,0  ,0,0,dxN4, -N4,0],
        [0,0,dyN4,0, -N4, 0,0,dyN4,0, -N4, 0,0,dyN4,0, -N4  ,0,0,dyN4,0, -N4]
        ])
    
    # Bs = np.matrix( [ 
    #     [0,0,dxN1, N1,0, 0,0,dxN2, N2,0, 0,0,dxN3, N3,0  ,0,0,dxN4, N4,0],
    #     [0,0,dyN1,0, N1, 0,0,dyN2,0, N2, 0,0,dyN3,0, N3  ,0,0,dyN4,0, N4],
        
    #     [0,0,dxN1, N1,0, 0,0,dxN2, N2,0, 0,0,dxN3, N3,0  ,0,0,dxN4, N4,0],
    #     [0,0,dyN1,0, N1, 0,0,dyN2,0, N2, 0,0,dyN3,0, N3  ,0,0,dyN4,0, N4],
        
    #     [0,0,dxN1, N1,0, 0,0,dxN2, N2,0, 0,0,dxN3, N3,0  ,0,0,dxN4, N4,0],
    #     [0,0,dyN1,0, N1, 0,0,dyN2,0, N2, 0,0,dyN3,0, N3  ,0,0,dyN4,0, N4],
        
    #     [0,0,dxN1, N1,0, 0,0,dxN2, N2,0, 0,0,dxN3, N3,0  ,0,0,dxN4, N4,0],
    #     [0,0,dyN1,0, N1, 0,0,dyN2,0, N2, 0,0,dyN3,0, N3  ,0,0,dyN4,0, N4]
    #     ])
    
    # Bs = np.matrix( [ 
    #     [0,0,dxN1, -N1,0, 0,0,dxN2, -N2,0, 0,0,dxN3, -N3,0  ,0,0,dxN4, -N4,0],
    #     [0,0,dyN1,0, -N1, 0,0,dyN2,0, -N2, 0,0,dyN3,0, -N3  ,0,0,dyN4,0, -N4],
        
    #     [0,0,dxN1, -N1,0, 0,0,dxN2, -N2,0, 0,0,dxN3, -N3,0  ,0,0,dxN4, -N4,0],
    #     [0,0,dyN1,0, -N1, 0,0,dyN2,0, -N2, 0,0,dyN3,0, -N3  ,0,0,dyN4,0, -N4],
        
    #     [0,0,dxN1, -N1,0, 0,0,dxN2, -N2,0, 0,0,dxN3, -N3,0  ,0,0,dxN4, -N4,0],
    #     [0,0,dyN1,0, -N1, 0,0,dyN2,0, -N2, 0,0,dyN3,0, -N3  ,0,0,dyN4,0, -N4],
        
    #     [0,0,dxN1, -N1,0, 0,0,dxN2, -N2,0, 0,0,dxN3, -N3,0  ,0,0,dxN4, -N4,0],
    #     [0,0,dyN1,0, -N1, 0,0,dyN2,0, -N2, 0,0,dyN3,0, -N3  ,0,0,dyN4,0, -N4]
    #     ])
    
    # print("Bs")
    # print(Bs)
    
    # Bs = np.matrix( [ 
    #     [0,0,dxN1,0 ,N1, 0,0,dxN2,0,N2, 0,0,dxN3,0,N3  ,0,0,dxN4,0,N4],
    #     [0,0,dyN1,-N1,0, 0,0,dyN2,-N2,0, 0,0,dyN3,-N3,0  ,0,0,dyN4,-N4,0]
    #     ])
    
    Bss = correc_Bs(xgs,ygs,x,y,Bs,jac_m_inv)
    
    return Bm,Bb,Bss,Jac_module

def correc_Bs(xgs,ygs,x,y,B,J_inv):
    
    #== Colocation Points:
    
    cx = np.array( [0,1,0,-1] ) /3**0.5
    
    cy = np.array([-1 , 0 , 1 ,  0 ]) /3**0.5
    
    # cx = np.array( [-1,1,1,-1] ) /3**0.5
    
    # cy = np.array([-1 ,-1 , 1 ,  1 ]) /3**0.5
    
    c     = np.zeros((8,8))

    b_bar = [];

    N    = np.zeros(4)
    dxN1 = np.zeros(4)
    dyN1 = np.zeros(4)
    
    xjacm = np.zeros((2,2))
    for i in range(4):
        dxN = np.zeros(4)
        dyN = np.zeros(4)
        N[0] = (1-cx[i])*(1-cy[i])/4
        N[1] = (1+cx[i])*(1-cy[i])/4
        N[2] = (1+cx[i])*(1+cy[i])/4 
        N[3] = (1-cx[i])*(1+cy[i])/4 
      
        dxN1[0] = (-1+cy[i])/4
        dxN1[1] = ( 1-cy[i])/4
        dxN1[2] = ( 1+cy[i])/4
        dxN1[3] = (-1-cy[i])/4
        
        dyN1[0] = (-1+cx[i])/4
        dyN1[1] = (-1-cx[i])/4
        dyN1[2] = ( 1+cx[i])/4
        dyN1[3] = ( 1-cx[i])/4
        
        # jac_m = np.dot(np.matrix( [[dxN1],[dyN1]] ) , np.matrix([x,y]))
        
        xjacm[0,0] = x[0]*dxN1[0] + x[1]*dxN1[1] + x[2]*dxN1[2] + x[3]*dxN1[3]
        xjacm[0,1] = y[0]*dxN1[0] + y[1]*dxN1[1] + y[2]*dxN1[2] + y[3]*dxN1[3]
        xjacm[1,0] = x[0]*dyN1[0] + x[1]*dyN1[1] + x[2]*dyN1[2] + x[3]*dyN1[3]
        xjacm[1,1] = y[0]*dyN1[0] + y[1]*dyN1[1] + y[2]*dyN1[2] + y[3]*dyN1[3]
        
        
        c[i*2:i*2+2,i*2:i*2+2] = xjacm
        jac_m_inv= np.linalg.inv(xjacm)
        # print()
        
        dxN[0] = jac_m_inv[0,0]*dxN1[0]+jac_m_inv[0,1]*dyN1[0];
        dxN[1] = jac_m_inv[0,0]*dxN1[1]+jac_m_inv[0,1]*dyN1[1];
        dxN[2] = jac_m_inv[0,0]*dxN1[2]+jac_m_inv[0,1]*dyN1[2];
        dxN[3] = jac_m_inv[0,0]*dxN1[3]+jac_m_inv[0,1]*dyN1[3];
        
        dyN[0] = jac_m_inv[1,0]*dxN1[0]+jac_m_inv[1,1]*dyN1[0];
        dyN[1] = jac_m_inv[1,0]*dxN1[1]+jac_m_inv[1,1]*dyN1[1];
        dyN[2] = jac_m_inv[1,0]*dxN1[2]+jac_m_inv[1,1]*dyN1[2];
        dyN[3] = jac_m_inv[1,0]*dxN1[3]+jac_m_inv[1,1]*dyN1[3];
               
        # 2x5
        bmat_s1  = np.matrix( [ [0,0, dxN[0], -N[0],    0  ],
                                [0,0, dyN[0],     0, -N[0] ] ])
                 
        bmat_s2  = np.matrix( [ [0,0, dxN[1], -N[1],    0  ],
                                [0,0, dyN[1],     0, -N[1] ] ])
                   
        bmat_s3  = np.matrix( [ [0,0, dxN[2], -N[2],    0  ],
                                [0,0, dyN[2],     0, -N[2] ] ])
                   
        bmat_s4  = np.matrix( [ [0,0, dxN[3],-N[3],    0  ],
                                [0,0, dyN[3],     0, -N[3] ] ])
        
        # bmat_s1  = np.matrix( [ [0,0, dxN[0], N[0],    0  ],
        #                         [0,0, dyN[0],     0, N[0] ] ])
                 
        # bmat_s2  = np.matrix( [ [0,0, dxN[1], N[1],    0  ],
        #                         [0,0, dyN[1],     0, N[1] ] ])
                   
        # bmat_s3  = np.matrix( [ [0,0, dxN[2], N[2],    0  ],
        #                         [0,0, dyN[2],     0, N[2] ] ])
                   
        # bmat_s4  = np.matrix( [ [0,0, dxN[3], N[3],    0  ],
        #                         [0,0, dyN[3],     0, N[3] ] ])

        # bmat_s1  = np.matrix( [ [0,0, dxN[0],     0,  N[0]  ],
        #                         [0,0, dyN[0],  -N[0],   0 ] ])
                 
        # bmat_s2  = np.matrix( [ [0,0, dxN[1], 0,N[1] ],
        #                         [0,0, dyN[1],    - N[1] ,0] ])
                   
        # bmat_s3  = np.matrix( [ [0,0, dxN[2], 0,N[2] ],
        #                         [0,0, dyN[2],    - N[2] ,0] ])
                   
        # bmat_s4  = np.matrix( [ [0,0, dxN[3], 0,N[3]  ],
        #                         [0,0, dyN[3],    - N[3],0 ] ])
        
        bmat_s = np.concatenate((bmat_s1,bmat_s2,bmat_s3,bmat_s4),axis=1)#ok 
        # bmat_s = np.concatenate((B,B,B,B),axis=0)#ok 
        
        if i ==0:
            b_bar = bmat_s
        else:
            # 8x24
            b_bar = np.concatenate( (b_bar,bmat_s),axis=0)
    
    
    T_mat = np.array ( [ [ 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],
                         [ 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 ],
                         [ 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 ],
                         [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 ] ] )
          
    P_mat = np.array ( [ [ 1 , -1 ,  0 ,  0 ], 
                             [0 ,  0 ,  1 ,  1 ],
                             [1 ,  1 ,  0 ,  0 ],
                             [0 ,  0 ,  1 , -1 ] ] )
          
    A_mat = np.array( [ [ 1 , ygs , 0 ,   0 ],
                        [ 0 ,   0 , 1 , xgs ] ])
    #  2x24 =      2x2     2x4         4x4                4x8   8x8  8x20
    # bmat_ss =  J_inv @ A_mat @ np.linalg.inv(P_mat) @ T_mat @ c @ B
    bmat_ss =  J_inv @ A_mat @ np.linalg.inv(P_mat) @ T_mat @ c @ b_bar
    # bmat_ss =  jac_m_inv @ A_mat @ np.linalg.inv(P_mat) @ T_mat @ c @ b_bar
    
    # print(bmat_ss)
    return bmat_ss

def coplanar_detect(normal, elem, coor,nne,nnod,nE):
    nodes_ = []
    node_det = []
    
    el_by_node = []
    el_det =[]
    
    nodes_by_el = []
    for i in range(nE):
        nodes_by_el.append([i])
    
    
    for k in range(nnod):
        nodes_.append([])
        el_by_node.append([])
        # nodes_by_el.append([])
        index = []
        value = []
        for i in range(nE):
            for j in range(nne):
                
                if elem[i][j] == k:
                    nodes_[k].append(normal[i])
                    el_by_node[k].append(i)
                    # nodes_by_el[i].append(k)
                    index.append(i)
                    value.append(j)
                    
        # print(value)  
        check= len(nodes_[k])
        not_pass = 0
        if  check >1:
            for l in range(1,check):
                if np.linalg.norm( np.cross( nodes_[k][l],nodes_[k][0] ) ) < 0.08:
                    # print(np.linalg.norm( np.cross( nodes_[k][l],nodes_[k][0] ) ))
                    not_pass = 1
            if not_pass==1:
                node_det.append(k)
                el_det = el_det + el_by_node[k]
                # print("elements")
                # print(index)   
                # print("nodes")
                # print(k)
                for i in range(len(index)):
                    # nodes_by_el[index[i]].append(k)
                    nodes_by_el[index[i]].append(value[i])

    # falta comparar las normales en cada nodo y definir el criterio de cuasi coplanar
    return node_det, list(set(el_det)), nodes_by_el

def springLAdd(K_mat, kL, springL_el, R_mat):
    KL = kL * np.matrix( [ [1,0,0,-1,0,0],
                           [0,0,0,0,0,0],
                           [0,0,0,0,0,0],
                           [-1,0,0,1,0,0],
                           [0,0,0,0,0,0],
                           [0,0,0,0,0,0]
                           ])
    # print( np.dot( np.dot(  np.transpose(R_mat)[:,:,0], KL) , R_mat))
    # temp = K_mat[springL_el,:,:]
    # print(temp[0])
    K_mat[springL_el,:,:] = np.dot( np.dot(  np.transpose(R_mat)[:,:,0], KL) , R_mat) # +  temp[0]
    return K_mat

def springRAdd(K_mat, kT, springT_el, R_mat):
    KL = kT * np.matrix( [ [0,0,0,0,0,0],
                           [0,0,0,0,0,0],
                           [0,0,1,0,0,-1],
                           [0,0,0,0,0,0],
                           [0,0,0,0,0,0],
                           [0,0,-1,0,0,1]
                           ])
    # temp =  K_mat[springT_el,:,:]
    K_mat[springT_el,:,:] = np.dot( np.dot(  np.transpose(R_mat)[:,:,0], KL) , R_mat) #+ temp[0]
    return K_mat

def springLAddR_mat(Le, coord,elem,el,R_mat):
    
    ve = np.zeros(2)
    # Le[i]   = np.linalg.norm(coord[elem[i,0],:]-coord[elem[i,1],:])
    ve= (coord[elem[el,][0][0],:]-coord[elem[el][0][1],:])/Le
    # print(ve)
    c1   = ve[0]
    c2   = ve[1]
    
    R_add = np.matrix([ [ c1 , c2 , 0 ,  0  , 0  , 0 ] , 
                        [ 0  , 0  , 0 ,  0  , 0  , 0 ] ,
                        [ 0  , 0  , 0 ,  0  , 0  , 0 ] ,
                        [ 0  , 0  , 0 ,  c1 , c2 , 0 ] ,
                        [ 0  , 0  , 0 ,  0  , 0  , 0 ] ,
                        [ 0  , 0  , 0 ,  0  , 0  , 0 ] 
                            ] )
    R_mat[el,:,:] = R_add
    
    return R_mat

def springRAddR_mat(Le, coord,elem,el,R_mat):
    
    ve = np.zeros(2)
    # Le[i]   = np.linalg.norm(coord[elem[i,0],:]-coord[elem[i,1],:])
    ve= (coord[elem[el,][0][0],:]-coord[elem[el][0][1],:])/Le
    # print(ve)
    c1   = ve[0]
    c2   = ve[1]
    
    R_add = np.matrix([       [ 0 , 0 , 0 , 0 , 0 , 0 ] , 
                              [ 0 , 0 , 0 , 0 , 0 , 0 ] ,
                              [ 0 , 0 , 1 , 0 , 0 , 0 ] ,
                              [ 0 , 0 , 0 , 0 , 0 , 0 ] ,
                              [ 0 , 0 , 0 , 0 , 0 , 0 ] ,
                              [ 0 , 0 , 0 , 0 , 0 , 1 ] 
                            ] )
    R_mat[el,:,:] = R_add
    
    return R_mat

def properties():
    return 

def Assembly2D(K_mat, elements, nndof, gl, nE):
    
    K_ = np.matrix(np.zeros([nndof,nndof]))
    
    for i in range(nE):
        j=gl*elements[i,0]
        l=gl*elements[i,1]
        
        K_[j:j+gl,j:j+gl] = K_mat[i,0:gl,0:gl] + K_[j:j+gl,j:j+gl]
        
        K_[l:l+gl,l:l+gl] = K_mat[i,gl:gl*2,gl:gl*2] + K_[l:l+gl,l:l+gl]
        
        K_[j:j+gl,l:l+gl] = K_mat[i,0:gl,gl:gl*2] + K_[j:j+gl,l:l+gl]
        
        K_[l:l+gl,j:j+gl] = K_mat[i,gl:gl*2,0:gl] + K_[l:l+gl,j:j+gl]
    return K_

def find_groups_nodes(Ltable, cells, fix,fix_value,nne):
    
    # fixcells_ = np.where (Ltable[:] !=-1)[0]
    fixnodes =[]
    apN = len(fix)-1
    
    indexes = np.where(Ltable==fix[0]) [0]
    # print(indexes)
    for i in range( len (indexes)):
        for j in range(apN):
            fixnodes.append( [indexes[i] , fix[j+1] , fix_value[j] ] )
    # print(fixnodes)
    fixnodes = np.array(fixnodes)
    
    fixnodes = fixnodes[fixnodes[:,0].argsort(),]
    return np.unique(fixnodes,axis=0)

def find_groups_nodes2(Ltable, cells, fix,fix_value,nne):
    
    # fixcells_ = np.where (Ltable[:] !=-1)[0]
    fixnodes =[]
    apN = len(fix)-1
    
    indexes_ = np.where(Ltable==fix[0]) [0]
    indexes =[]
    for i in range(len(indexes_)):
        indexes = list(set(indexes +cells[indexes_[i]][1:3]))
    # print()
    # print(indexes_)
    for i in range( len (indexes)):
        
        for j in range(apN):
            fixnodes.append( [indexes[i] , fix[j+1] , fix_value[j] ] )
    # print(fixnodes)
    fixnodes = np.array(fixnodes)
    
    fixnodes = fixnodes[fixnodes[:,0].argsort(),]
    return np.unique(fixnodes,axis=0)

def find_groups_elems(Ltable, cells, propr,nne,nE):  
    
    elems =[]
    ele = 0
    # if propr[0] !=0:
            
    for i in range( np.shape(Ltable)[0]):
        if cells[i][0] == nne:
            if int(Ltable[i]) ==propr:
                elems.append(ele)            
            ele = ele+1
    # elif propr[0] ==0:
    #     elems = list(range(nE))
    return elems

def properties_beam(Ltable, cells, Ae , Ee , Ie , nE , prop,nne):
    indexs = find_groups_elems(Ltable,cells,prop[0],nne,nE)
    Ae[indexs] = prop[1]
    Ee[indexs] = prop[2]
    Ie[indexs] = prop[3]
    return Ae, Ee, Ie

def properties_beam2(Ltable, cells, Ae , Ee , Ie , Rhoe, lumped_e, nE , prop,nne):
    indexs = find_groups_elems(Ltable,cells,prop[0],nne,nE)
    Ae[indexs] = prop[1]
    Ee[indexs] = prop[2]
    Ie[indexs] = prop[3]
    Rhoe[indexs] = prop[4]
    lumped_e[indexs] = prop[5]
    return Ae, Ee, Ie, Rhoe, lumped_e

def properties_beam_3d(Ltable, cells, nE , prop,nne):
    Ae    = np.ones(nE) 
    Iye   = np.ones(nE) 
    Ize   = np.ones(nE)
    Je    = np.ones(nE) 
    Ee    = np.ones(nE) 
    Ge    = np.ones(nE) 
    ori_e = np.ones((nE,3)) 
    
    for i in range(len(prop)):
        indexs = find_groups_elems(Ltable,cells,prop[i][0],nne,nE)
        Ae[indexs]    = prop[i][1]
        Ee[indexs]    = prop[i][2]
        Ge[indexs]    = prop[i][3]
        Iye[indexs]   = prop[i][4]
        Ize[indexs]   = prop[i][5]
        Je[indexs]    = prop[i][6]
        ori_e[indexs,] = prop[i][7]
    return Ae, Ee, Iye, Ize, Je, Ee, Ge, ori_e

def properties_beam_3d_dyn(Ltable, cells, nE , prop,nne):
    Ae    = np.ones(nE) 
    Iye   = np.ones(nE) 
    Ize   = np.ones(nE)
    Je    = np.ones(nE) 
    Ee    = np.ones(nE) 
    Ge    = np.ones(nE) 
    ori_e = np.ones((nE,3)) 
    Rho_e = np.ones(nE) 

    for i in range(len(prop)):
        indexs = find_groups_elems(Ltable,cells,prop[i][0],nne,nE)
        Ae[indexs]    = prop[i][1]
        Ee[indexs]    = prop[i][2]
        Ge[indexs]    = prop[i][3]
        Iye[indexs]   = prop[i][4]
        Ize[indexs]   = prop[i][5]
        Je[indexs]    = prop[i][6]
        ori_e[indexs,] = prop[i][7]
        Rho_e[indexs]    = prop[i][8]
    return Ae, Ee, Iye, Ize, Je, Ee, Ge, ori_e, Rho_e


def properties_Truss(Ltable, cells, Ae , Ee , nE , prop,nne):
    indexs = find_groups_elems(Ltable,cells,prop[0],nne, nE)
    Ae[indexs] = prop[1]
    Ee[indexs] = prop[2]
    return Ae, Ee

def properties_Truss_2(Ltable, cells, Ae , Ee , rhoe, lump_e, nE , prop,nne):
    indexs = find_groups_elems(Ltable,cells,prop[0],nne, nE)
    Ae[indexs]   = prop[1]
    Ee[indexs]   = prop[2]
    rhoe[indexs] = prop[3]
    lump_e[indexs] = prop[4]
    return Ae, Ee, rhoe, lump_e

def properties_shell(Ltable, cells, Ee, poi_e, thick_e,denss_e , nE , prop,nne):
    for i in range(len(prop)):
        indexs = find_groups_elems(Ltable,cells,prop[i][0],nne,nE)
        Ee[indexs]      = prop[i][1]
        poi_e[indexs]   = prop[i][2]
        thick_e[indexs] = prop[i][3]
        denss_e[indexs] = prop[i][4]
    return Ee, poi_e, thick_e,denss_e

# def centroid_(coord, elems, Ltable, cells, nE , prop,nne):
#     centroid_e =[]
#     for i in range(len(prop)):
#         indexs = find_groups_elems(Ltable,cells,prop[i][0],nne,nE)
#         print(indexs)
#         for j in range(len(indexs)):
#             centroid_e.append( np.array([ np.mean(coord[elems[indexs[j]],0]),np.mean(coord[elems[indexs[j]],1]) ,np.mean(coord[elems[indexs[j]],2])  ])  )
#     return centroid_e

def centroid_(coord, elems, Ltable, cells, nE , prop,nne,centroid_e):
    # centroid_e =[]
    for i in range(len(prop)):
        indexs = find_groups_elems(Ltable,cells,prop[i][0],nne,nE)
        # print(indexs)
        for j in range(len(indexs)):
            centroid_e[indexs[j],:]= np.array([ np.mean(coord[elems[indexs[j]],0]),np.mean(coord[elems[indexs[j]],1]) ,np.mean(coord[elems[indexs[j]],2])  ])  
    return centroid_e

def ref_point(Ltable, cells, centroid_,ref_point_,ref,prop,nne, nE):
    for i in range(len(prop)):
        indexs = find_groups_elems(Ltable,cells,prop[i][0],nne,nE)
        # print(indexs)
        for j in range(len(indexs)):
            if np.size(ref)==3:
                ref_point_[indexs[j],:] = centroid_[indexs[j],:] + ref
            else:
                ref_point_[indexs[j],:] = centroid_[indexs[j],:] + ref[indexs[j],:]
    return ref_point_

def SysRedu (nndof, fix_indices, K_, f, u):
    index_bool = np.ones(shape=(nndof,),dtype=bool)
    index_bool[fix_indices]=False

    index_redu = np.argwhere(index_bool==True)[:,0]

    K_redu_ = np.delete(K_,fix_indices,0)
    K_redu = np.delete(K_redu_,fix_indices,1)

    K_bc = np.delete(K_redu_,index_redu,1)

    f_redu = np.delete(f,fix_indices,0) -np.dot( K_bc , u[fix_indices])

    return K_redu, f_redu, index_redu
        
def displacement_e(elements, u , R_mat,nE,nne,gl):
    u_e = np.zeros((nE,nne*gl,1))

    for i in range(nE):
        i1 = elements[i,0]*gl
        i2 = elements[i,1]*gl
        index = list(range(i1,i1+gl))+list(range(i2,i2+gl))
        # print(index)
        u_e[i,:,] = np.dot (  (R_mat[i,:,:]), u[index])
    return u_e

def apply_u(fixnodes,u,gl):
    fixnodeslen = np.shape(fixnodes)[0]
    fix_indices = np.zeros(fixnodeslen, int)
    for i in range(fixnodeslen):
        ieqn = (fixnodes[i,0])*gl + fixnodes[i,1] - 1   # Find equation number
        # print(ieqn)
        # print(ieqn)
        u[int(ieqn),] = fixnodes[i,2]                              # and store the solution in u
        fix_indices[i] = ieqn   
    return u, fix_indices

def apply_fT(nE, nne , gl ,f, Ae , Ee, R_mat, alf_e, index, delT_e):
    f_ter = np.zeros((nE,nne*gl,1))
    for j in range(len(index)):
        i = index[j]
        temp = (alf_e[i]*Ae[i]*Ee[i]*delT_e[i]* np.dot( np.transpose(R_mat[i,:,:]) , np.array([ [-1],[0],[0],[1],[0],[0] ])) )
        f_ter[i,:,] = temp
    return f_ter

def apply_fq1(nE, nne , gl ,f, Le , Ee, R_mat, index, loadq):
    f_q = np.zeros((nE,nne*gl,1))
    for j in range(len(index)):
        i = index[j]
        temp = loadq[0] * Le[i] * np.array( [0.0, 0.5, 1/12*Le[i], 0.0 ,0.5, -1/12*Le[i]])
        f_q[i,:,0] = temp
    return f_q

def AssemblyF(nE , elements, gl, f, fN= None, fT = None, fq =None):
    if fT is not None:
        for i in range(nE):
            j=gl*elements[i,0]
            l=gl*elements[i,1]
            f[j:j+gl,] = fT[i,:3,] + f[j:j+gl,]
            f[l:l+gl,] = fT[i,3:,] + f[l:l+gl,]
    if fN is not None:
         f = fN +f
    if fq is not None:
        for i in range(nE):
            j=gl*elements[i,0]
            l=gl*elements[i,1]
            f[j:j+gl,] = fq[i,:3,] + f[j:j+gl,]
            f[l:l+gl,] = fq[i,3:,] + f[l:l+gl,]
    return f

def interpolation_beam2D (n_int, nE, coordinates, elements, u, u_e, Le, R_mat):
    
    # n_int = 10
    r     = np.linspace(-1,1,n_int,endpoint=False)
    x_new = np.zeros((nE,n_int))
    y_new = np.zeros((nE,n_int))
    theta_new = np.zeros((nE,n_int))
    
    
    coord_inter = np.copy(coordinates)
    
    for i in range(nE):
        
        j=elements[i,0]
        l=elements[i,1]
        
        x1 = coordinates[j,0,]
        x2 = coordinates[l,0,]
        
        y1 = coordinates[j,1,]
        y2 = coordinates[l,1,]
        
        xnews = np.linspace(x1,x2,n_int,endpoint=False)
        ynews = np.linspace(y1,y2,n_int,endpoint=False)
                
        coord_inter = np.concatenate( (coord_inter ,np.transpose (np.array([xnews[1:],ynews[1:]])) ) ,axis=0)

    N1 = (1-r )/2
    N2 = (1+r )/2
    
    H1 = 1/4*(1-r)**2*(2+r)
    H2 = 1/4*(1-r)**2*(1+r)
    H3 = 1/4*(1+r)**2*(2-r)
    H4 = 1/4*(1+r)**2*(r-1)
    
    H1p = 3/4*(-1 + r**2)
    H2p = 1/4*(-1 - 2*r + 3*r**2)
    H3p = 3/4*( 1 - r**2)
    H4p = 1/4*(-1 + 2*r + 3*r**2)
    
    u_new = np.copy(u)
    
    for i in range(nE):
    
        x_new[i,:,] = N1*u_e[i,0,] + N2*u_e[i,3,]
        
        y_new[i,:,] = H1* u_e[i,1,] + H2*Le[i]/2 * u_e[i,2,]+ H3* u_e[i,4,]+ H4*Le[i]/2* u_e[i,5,]
        
        theta_new[i,:,] = H1p*2/Le[i] * u_e[i,1,] + H2p * u_e[i,2,] + H3p * 2/Le[i] * u_e[i,4,] + H4p*u_e[i,5,]
        
        for j in range(1,n_int):
            temp = np.array([ x_new[i,j,], y_new[i,j,], theta_new[i,j,] ] )
            u_new = np.append (u_new,  np.dot( np.transpose ( R_mat[i,:3,:3]), temp ) )
            # if i==2:
                # print(temp)
    u_new = u_new.reshape(( np.shape(u_new)[0],1)) 
    
    ## falta implementar la creacion de una nueva tabla de elementos
    
    return coord_inter, u_new

def addCoor(coord,umov,npnod,gl):
    index1 = list(range(0,npnod*gl,gl))
    index2 = list(range(1,npnod*gl+1,gl))
    
    coord_new = np.copy(coord)
    coord_new[:,0] = coord[:,0] + umov[index1][:,0]
    coord_new[:,1] = coord[:,1] + umov[index2][:,0]
    
    return coord_new

def apply_NVMe(ue,R,K):
    NVM = np.zeros(np.shape(ue))
    for i in range(np.shape(ue)[0]):
        k = np.dot( np.dot( R[i,:,:] , K[i,:,:]) ,  np.transpose(R[i,:,:]))
        NVM[i,:,:] = np.dot(k,ue[i,:,:])
    return NVM

def apply_K_transf(K_,node,ang,glG):
    T_new = np.identity(np.shape(K_)[0])
    i = node*glG
    T_new[i:i+glG,i:i+glG] = np.matrix( [ [np.cos(ang) , np.sin(ang)],[-np.sin(ang),np.cos(ang)]])
    
    K_ = np.dot(T_new, np.dot(K_,np.transpose(T_new)))
    # print(T_new)
    return K_

def apply_K_transf2(u,node,ang,glG):
    T_new = np.identity(np.shape(u)[0])
    i = node*glG
    T_new[i:i+glG,i:i+glG] = np.matrix( [ [np.cos(ang) , -np.sin(ang)],[np.sin(ang),np.cos(ang)]])
    
    K_ = np.dot(T_new, u)
    # print(T_new)
    return K_

def stressEuler2D(npnod, nE, nne, gl,glG, u, elements, R_mat, Le, Ee, ymax, ymin):  
    g1 = -1/3**0.5
    g2 = 1/3**0.5
    
    u_e = displacement_e(elements, u , R_mat,nE,nne,gl)
    
    Stress = np.zeros( (npnod,5) )
    average = np.zeros(npnod )
    
    for i in range(nE):
        i1 = elements[i,0]
        i2 = elements[i,1]
        
        # Axial 
        StressA  = Ee[i] * ( -1/Le[i]*u_e[i,1,] + 1/Le[i]*u_e[i,3,] )
        
        #bending
        # gauss point 1
        bmat_1 =  np.array ( [0,6*g1/Le[i]**2 , (-1+3*g1)/Le[i] , 0, -6*g1/Le[i]**2 , (1+3*g1)/Le[i] ])
        # Mb = E*Ie[i]* np.dot(bmat_1,np.transpose( u_e[i,:,]) )
        StressB1 = ymax*Ee[i]* np.dot(bmat_1,( u_e[i,:,]) )
        # gauss point 2
        bmat_2 =  np.array ( [0,6*g2/Le[i]**2 , (-1+3*g2)/Le[i] , 0, -6*g2/Le[i]**2 , (1+3*g2)/Le[i] ])
        StressB2 = ymax*Ee[i]* np.dot(bmat_2,( u_e[i,:,]) )
        
        Stress[i1,0] = StressA + Stress[i1,0]
        Stress[i2,0] = StressA + Stress[i2,0]
        
        Stress[i1,1] = StressB1 + Stress[i1,1]
        Stress[i2,1] = StressB2 + Stress[i2,1]
        
        average[i1] = average[i1] + 1
        average[i2] = average[i2] + 1
        
    Stress[:,0] = Stress[:,0]/average
    Stress[:,1] = Stress[:,1]/average
    Stress[:,2] = -Stress[:,1]/ymax*ymin
    Stress[:,3] = Stress[:,0] + np.abs ( Stress[:,2])
    Stress[:,4] = Stress[:,0] - np.abs ( Stress[:,2])
    
    return Stress

def Section_calc(prop,nE,Ltable,cells,nne):
    y = np.zeros(nE)
    Q = np.zeros(nE)
    b = np.zeros(nE)
    for i in range(len(prop)):
        index = find_groups_elems(Ltable, cells, prop[i][0], nne)
        if prop[i][1] == "Rectangular":
            y[index] = prop[i][3]/2
            b[index] = prop[i][2]
            Q[index] = prop[i][2]*prop[i][3]**2/6
        elif prop[i][1] == "Redonda":
            y[index] = prop[i][2]/2
            b[index] = prop[i][2]
            Q[index] = np.pi*prop[i][2]**3/32
        elif prop[i][1] == "Caño Rectangular":
            y[index] = prop[i][3]/2
            b[index] = prop[i][2]
            Q[index] = prop[i][2]*prop[i][3]**2/6 - (prop[i][2]-2*prop[i][4])*(prop[i][3]-2*prop[i][4])**2/6
    return Q,b,y

def stressEuler2D_v2(Ltable, cells,elements,npnod, nE, nne, glG, R_mat, F_e, prop_beam):  
       
    Stress = np.zeros( (npnod,3) )
    average = np.zeros(npnod )
    
    Ae, Ee, Le ,Ie,sect_prop = prop_beam
    
    Qe , be , ye_ = Section_calc(sect_prop,nE,Ltable,cells,nne)
    # print(Qe)
    
    for i in range(nE):
        i1 = elements[i,0]
        i2 = elements[i,1]
        
        # Axial 
        StressA  = 1/Ae[i]* ( -F_e[i,0,] + F_e[i,3,] )
        
        # Bending
        
        if Ie[i] == 0:
        
            StressB_n1 = 0 
            StressB_n2 = 0
        # Shear
        
            Tau_n1 = 0
            Tau_n2 = 0    
        
        else:    
            StressB_n1 = -1/Ie[i]*F_e[i,2,]*ye_[i] 
            
            StressB_n2 = 1/Ie[i]*F_e[i,5,]*ye_[i] 
        
            Tau_n1 = F_e[i,1,] *Qe[i]/Ie[i]/be[i]
            Tau_n2 = -F_e[i,4,] *Qe[i]/Ie[i]/be[i]
        
        # Build in nodes
        Stress[i1,0] = StressA + Stress[i1,0]
        Stress[i2,0] = StressA + Stress[i2,0]
        
        Stress[i1,1] = StressB_n1 + Stress[i1,1]
        Stress[i2,1] = StressB_n2 + Stress[i2,1]
        
        Stress[i1,2] = Tau_n1 + Stress[i1,2]
        Stress[i2,2] = Tau_n2 + Stress[i2,2]
        
        average[i1] = average[i1] + 1
        average[i2] = average[i2] + 1
        
    Stress[:,0] = Stress[:,0]/average
    Stress[:,1] = Stress[:,1]/average
    Stress[:,2] = Stress[:,2]/average
    
    
    return Stress

def apply_p_cte(coord, elem,glG,f, group, nne,p,ref_p):
    
    gauss_x = np.zeros(4)
    gauss_y = np.zeros(4)
    gauss_w = np.zeros(4)
    
    gauss_x[0] = -1/np.sqrt(3)
    gauss_y[0] = -1/np.sqrt(3)
    gauss_w[0] = 1
    
    gauss_x[1] =  1/np.sqrt(3)
    gauss_y[1] = -1/np.sqrt(3)
    gauss_w[1] = 1
    
    gauss_x[2] =  1/np.sqrt(3)
    gauss_y[2] =  1/np.sqrt(3)
    gauss_w[2] = 1
    
    gauss_x[3] = -1/np.sqrt(3)
    gauss_y[3] =  1/np.sqrt(3)
    gauss_w[3] = 1
    
    t = np.array([0,0,-p],dtype=np.float64)
    # print(group)
    for i in range(len(group)):
        cxyz = coord[ elem[ group[i], ],:]#      % Element coordinates  
        # Transformacion
        Le, normal = Rotation_RShell(cxyz[0:3,:],ref_p)
        # coordenadas locales
        ctxy = np.zeros((4,3))
        for l in range(4):
            ctxy[l,:] = np.dot(Le[:3,:3] ,  cxyz[l,:])#    % Rotate coordinates to element mid plane
        # print(ctxy)
        # print(np.dot(np.transpose(Le[:3,:3]),t))
        if np.linalg.norm(ref_p - cxyz[0,:]) > np.linalg.norm(ref_p - cxyz[0,:] + normal):
            t_=-t
        else:
            t_ =t
        
        force = np.zeros((12,1 ))
        for j1 in range(int(nne)):
            Ni , Jac_module = N_force_surf(gauss_x[j1],gauss_y[j1],ctxy[:,0],ctxy[:,1])
            # print(Jac_module)
            for j3 in range(nne):
                # print(Le[2,2])
                force[3*j3:3*j3+3,] = Ni[j3]*np.array([[1,0,0],[0,1,0],[0,0,1]])*np.transpose( np.dot ( np.transpose(Le[:3,:3]) , t_)   )*Jac_module + force[3*j3:3*j3+3,]
                # force[j3*3:j3*3+3,] = Ni[j3] *t_.reshape(3,1) *Le[2,2] *Jac_module + force[j3*3:j3*3+3,]
                # 12x1
        # print(force)
        i1 = glG*elem[ group[i],0]
        i2 = glG*elem[ group[i],1]
        i3 = glG*elem[ group[i],2]
        i4 = glG*elem[ group[i],3]
        
        index_1 = [0,3,6,9]
        index_2 = [i1,i2,i3,i4]
        
        for j4 in range(4): 
            a = index_2[j4]
            b = index_1[j4]
            
            f[a:a+3,] =  f[a:a+3,] + force[b:b+3] 
                
    return f

def N_force_surf(xgs,ygs,x,y):
    
    dxN1  = np.zeros(4)
    dyN1  = np.zeros(4)
    jac_m = np.zeros((2,2))
    jac_m = np.asmatrix(jac_m)
    
    # Derivadas de las funciones de interpolacion
    dxN1[0] = (-1+ygs)/4
    dxN1[1] = ( 1-ygs)/4
    dxN1[2] = ( 1+ygs)/4
    dxN1[3] = (-1-ygs)/4
    
    dyN1[0] = (-1+xgs)/4
    dyN1[1] = (-1-xgs)/4
    dyN1[2] = ( 1+xgs)/4
    dyN1[3] = (1-xgs)/4
    
    # Jacobiano
    
    # jac_m = np.dot(np.matrix( [[dxN1],[dyN1]] ) , np.matrix([x,y]))
    jac_m[0,0] = x[0]*dxN1[0] + x[1]*dxN1[1] + x[2]*dxN1[2] + x[3]*dxN1[3];
    jac_m[0,1] = y[0]*dxN1[0] + y[1]*dxN1[1] + y[2]*dxN1[2] + y[3]*dxN1[3];
    jac_m[1,0] = x[0]*dyN1[0] + x[1]*dyN1[1] + x[2]*dyN1[2] + x[3]*dyN1[3];
    jac_m[1,1] = y[0]*dyN1[0] + y[1]*dyN1[1] + y[2]*dyN1[2] + y[3]*dyN1[3];
    
    jac_m_inv = np.linalg.inv(jac_m)
    
    Jac_module = np.abs(np.linalg.det( jac_m))
    
    Ni    = np.zeros(4)
    Ni[0]  = (1-xgs)*(1-ygs)/4 
    Ni[1]  = (1+xgs)*(1-ygs)/4 
    Ni[2]  = (1+xgs)*(1+ygs)/4 
    Ni[3]  = (1-xgs)*(1+ygs)/4 
    
    return Ni, Jac_module

def apply_NVM(F_e, coordinates, elements, elems):
    
    """    
    E = len(elems)
    
    N = np.zeros(E)
    V = np.zeros(E)
    M = np.zeros(E)
    x = np.zeros(E)
    
    array = elements[elems].flatten()
    len_array = np.unique(elements[elems].flatten())
    
    N_ = np.zeros(2*E)
    V_ = np.zeros(2*E)
    M_ = np.zeros(2*E)
    index1 =  list( range( 0,2*E,2))
    index2 =  list( range( 1,2*E,2))
    
    N_ [index1] = F_e[elems,0,0]
    N_ [index2] = F_e[elems,3,0]
    
    V_ [index1] = F_e[elems,1,0]
    V_ [index2] = F_e[elems,4,0]
    
    M_ [index1] = F_e[elems,2,0]
    M_ [index2] = F_e[elems,5,0]

    for i in range(E):
        sign = 1
        avg  = 0
        for j in range(np.size(array)):
            if array[j] == len_array[i]:
                N[i] = sign* N_[j] + N[i] 
                V[i] = sign* V_[j] + V[i] 
                M[i] = sign* M_[j] + M[i]                 
                sign = -1
                avg+=1
        N[i] = N[i]/avg
        V[i] = V[i]/avg
        M[i] = M[i]/avg
                              
    return N,V,M,x
    """
    return 1  

def Modes1(K,M):
    A = np.linalg.inv(M)@ K
    
    values , vectors = np.linalg.eig(A)
    
    return values, vectors

def Modes2(K,M):
    A = np.linalg.inv( np.array(M)**.5)@ K @ np.linalg.inv( np.array(M)**.5)
    
    values , vectors = np.linalg.eig(A)
    
    return values, vectors

def Modes3(K,M):
    A = np.linalg.inv( np.array(M)**.5)@ K @ np.linalg.inv( np.array(M)**.5)
    
    values , vectors = np.linalg.eig(A)
    
    return values, vectors