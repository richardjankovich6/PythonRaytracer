import numpy as np

from rayTracer import Point, Ray, DirectionalLight, Camera, Object, Sphere, Triangle, RayTracer

maxDepth = 3
at = np.float32

def tracePurpleSphere(width = 1920, height = 1080):
    dirToLight = np.array([1.0,1.0,1.0], at)
    lightColor = np.array([1.0,1.0,1.0], at)
    aLight = np.array([0.1,0.1,0.1], at)
    bgColor = np.array([0.2,0.2,0.2], at)
    light = DirectionalLight(dirToLight=dirToLight, lightColor=lightColor, aLight=aLight, bgColor=bgColor)

    lookAt=np.array([0.0,0.0,0.0], at)
    lookFrom=np.array([0.0,0.0,1.0], at)
    lookUp=np.array([0.0,1.0,0.0], at)
    camera = Camera(width=width, height=height, lookAt=lookAt, lookFrom=lookFrom, lookUp=lookUp, fov=90)
    
    purpleSphere = Sphere(center=np.array([0.0,0.0,0.0], at), dcolor=np.array([1.0,0.0,1.0], at), scolor=np.array([1.0,1.0,1.0], at), radius=0.4, kd=0.7, ks=0.2, ka=0.1, kgls=16.0, light=light)
    rayTracer = RayTracer(camera, light, [purpleSphere], width=width, height=height)
    rayTracer.traceToPPM("purpleSphereImageTest")

def traceMultiSpheres(width = 1920, height = 1080):
    dirToLight = np.array([1.0,1.0,1.0], at)
    lightColor = np.array([1.0,1.0,1.0], at)
    aLight = np.array([0.1,0.1,0.1], at)
    bgColor = np.array([0.2,0.2,0.2], at)
    light = DirectionalLight(dirToLight=dirToLight, lightColor=lightColor, aLight=aLight, bgColor=bgColor)

    lookAt=np.array([0.0,0.0,0.0], at)
    lookFrom=np.array([0.0,0.0,1.0], at)
    lookUp=np.array([0.0,1.0,0.0], at)
    camera = Camera(width=width, height=height, lookAt=lookAt, lookFrom=lookFrom, lookUp=lookUp, fov=90)
    
    whiteSphere = Sphere(center=np.array([0.45,0.0,-0.15], at), dcolor=np.array([1.0,1.0,1.0], at), scolor=np.array([1.0,1.0,1.0], at), radius=0.15, kd=0.8, ks=0.1, ka=0.3, kgls=4.0, light=light)
    whiteSphere = Sphere(center=np.array([0.45,0.0,-0.15], at), dcolor=np.array([1.0,1.0,1.0], at), scolor=np.array([1.0,1.0,1.0], at), radius=0.15, kd=0.6, ks=0.1, ka=0.3, kgls=4.0, light=light)
    redSphere = Sphere(center=np.array([0.0,0.0,-0.1], at), dcolor=np.array([1.0,0.0,0.0], at), scolor=np.array([1.0,1.0,1.0], at), radius=0.2, kd=0.6, ks=0.3, ka=0.1, kgls=32.0, light=light)
    greenSphere = Sphere(center=np.array([-0.6,0.0,0.0], at), dcolor=np.array([0.0,1.0,0.0], at), scolor=np.array([0.5,1.0,0.5], at), radius=0.3, kd=0.7, ks=0.2, ka=0.1, kgls=64.0, light=light)
    blueSphere = Sphere(center=np.array([0.0,-10000.5,0.0], at), dcolor=np.array([0.0,0.0,1.0], at), scolor=np.array([1.0,1.0,1.0], at), radius=10000.0, kd=0.9, ks=0.0, ka=0.1, kgls=16.0, light=light)
    rayTracer = RayTracer(camera, light, [whiteSphere, redSphere, greenSphere, blueSphere], width, height)
    rayTracer.traceToPPM("testImage3")

def traceScene1(width = 1920, height = 1080):
    dirToLight = np.array([0.0,1.0,0.0], at)
    lightColor = np.array([1.0,1.0,1.0], at)
    aLight = np.array([0.0,0.0,0.0], at)
    bgColor = np.array([0.2,0.2,0.2], at)
    light = DirectionalLight(dirToLight=dirToLight, lightColor=lightColor, aLight=aLight, bgColor=bgColor)

    lookAt=np.array([0.0,0.0,0.0], at)
    lookFrom=np.array([0.0,0.0,1.0], at)
    lookUp=np.array([0.0,1.0,0.0], at)
    camera = Camera(width=width, height=height, lookAt=lookAt, lookFrom=lookFrom, lookUp=lookUp, fov=90)

    reflectiveSphere = Sphere(center=np.array([0.0,0.3,-1.0], at), radius=0.25, dcolor=np.array([0.75,0.75,0.75], at), scolor=np.array([1.0,1.0,1.0], at), kd=0.0, ks=0.1, ka=0.1, kgls=10.0, refl=0.9, light=light)
    blueTriangle = Triangle(p1=np.array([0.0,-0.7,-0.5], at), p2=np.array([1.0,0.4,-1.0], at), p3=np.array([0.0,-0.7,-1.5], at), dcolor=np.array([0.0,0.0,1.0], at), scolor=np.array([1.0,1.0,1.0], at), kd=0.4, ks=0.5, ka=0.1, kgls=4.0, refl=0.0, light=light)
    yellowTriangle = Triangle(p1=np.array([0.0,-0.7,-0.5], at), p2=np.array([0.0,-0.7,-1.5], at), p3=np.array([-1.0,0.4,-1.0], at), dcolor=np.array([1.0,1.0,0.0], at), scolor=np.array([1.0,1.0,1.0], at), kd=0.9, ks=1.0, ka=0.1, kgls=4.0, refl=0.0, light=light)

    rayTracer = RayTracer(camera, light, [reflectiveSphere, blueTriangle, yellowTriangle], width, height)
    rayTracer.traceToPPM("scene1.4")
    
def traceScene2(width = 1920, height = 1080):
    lookAt=np.array([0.0,0.0,0.0], at)
    lookFrom=np.array([0.0,0.0,1.0], at)
    lookUp=np.array([0.0,1.0,0.0], at)
    camera = Camera(width=width, height=height, lookAt=lookAt, lookFrom=lookFrom, lookUp=lookUp, fov=90)

    dirToLight = np.array([1.0,0.0,0.0], at)
    lightColor = np.array([1.0,1.0,1.0], at)
    aLight = np.array([0.1,0.1,0.1], at)
    bgColor = np.array([0.2,0.2,0.2], at)
    light = DirectionalLight(dirToLight=dirToLight, lightColor=lightColor, aLight=aLight, bgColor=bgColor)

    whiteSphere = Sphere(center=np.array([0.5,0.0,-0.15], at), radius=0.05, dcolor=np.array([1.0,1.0,1.0], at), scolor=np.array([1.0,1.0,1.0], at), kd=0.8, ks=0.1, ka=0.3, kgls=4.0, refl=0.0, light=light)
    redSphere = Sphere(center=np.array([0.3,0.0,-0.1], at), radius=0.08, dcolor=np.array([1.0,0.0,0.0], at), scolor=np.array([0.5,1.0,0.5], at), kd=0.8, ks=0.8, ka=0.1, kgls=32.0, refl=0.0, light=light)
    greenSphere = Sphere(center=np.array([-0.6,0.0,0.0], at), radius=0.3, dcolor=np.array([0.0,1.0,0.0], at), scolor=np.array([0.5,1.0,0.5], at), kd=0.7, ks=0.5, ka=0.1, kgls=64.0, refl=0.0, light=light)
    reflectiveSphere = Sphere(center=np.array([0.1,-0.55,0.15], at), radius=0.3, dcolor=np.array([0.75,0.75,0.75], at), scolor=np.array([1.0,1.0,1.0], at), kd=0.0, ks=0.1, ka=0.1, kgls=10.0, refl=0.9, light=light)
    blueTriangle = Triangle(p1=np.array([0.3,-0.3,-0.4], at), p2=np.array([0.0,0.3,-0.1], at), p3=np.array([-0.3,-0.3,0.2], at), dcolor=np.array([0.0,0.0,1.0], at), scolor=np.array([1.0,1.0,1.0], at), kd=0.5, ks=0.5, ka=0.0 , kgls=32.0, refl=0.0, light=light)
    yellowTriangle = Triangle(p1=np.array([-0.2,0.1,0.1], at), p2=np.array([-0.2,-0.5,0.2], at), p3=np.array([-0.2,0.1,-0.3], at), dcolor=np.array([1.0,1.0,0.0], at), scolor=np.array([1.0,1.0,1.0], at), kd=0.9, ks=0.5, ka=0.1, kgls=4.0, refl=0.0, light=light)
    
    rayTracer = RayTracer(camera, light, [whiteSphere, redSphere, greenSphere, reflectiveSphere, blueTriangle, yellowTriangle], width, height)
    rayTracer.traceToPPM("scene2.7")

def traceCustomScene(width = 1920, height = 1080):
    lookAt=np.array([0.0,0.0,0.0], at)
    lookFrom=np.array([0.0,0.0,1.0], at)
    lookUp=np.array([0.0,1.0,0.0], at)
    camera = Camera(width=width, height=height, lookAt=lookAt, lookFrom=lookFrom, lookUp=lookUp, fov=90)

    dirToLight = np.array([1.0,1.0,0.0], at)
    lightColor = np.array([1.0,1.0,1.0], at)
    aLight = np.array([0.2,0.2,0.2], at)
    bgColor = np.array([0.2,0.2,0.2], at)
    light = DirectionalLight(dirToLight=dirToLight, lightColor=lightColor, aLight=aLight, bgColor=bgColor)

    blueSphere = Sphere(center=np.array([0.5,-0.1,-0.15], at), radius=0.25, dcolor=np.array([0.0,0.0,1.0], at), scolor=np.array([0.1,0.1,1.0], at), kd=0.8, ks=0.1, ka=0.1, kgls=2.0, refl=0.0, light=light)
    redSphere = Sphere(center=np.array([0.2,0.1,-0.1], at), radius=0.18, dcolor=np.array([1.0,0.0,0.0], at), scolor=np.array([1.0,0.1,0.1], at), kd=0.2, ks=0.7, ka=0.1, kgls=12.0, refl=0.0, light=light)
    greenSphere = Sphere(center=np.array([-0.6,0.0,0.0], at), radius=0.3, dcolor=np.array([0.0,1.0,0.0], at), scolor=np.array([0.1,1.0,0.1], at), kd=0.4, ks=0.5, ka=0.1, kgls=20.0, refl=0.0, light=light)
    # reflectiveSphere = Sphere(center=np.array([0.1,-0.55,0.15], at), radius=0.3, dcolor=np.array([0.75,0.75,0.75], at), scolor=np.array([1.0,1.0,1.0], at), kd=0.0, ks=0.1, ka=0.1, kgls=10.0, refl=0.9, light=light)
    
    purpleTriangle = Triangle(p1=np.array([0.3,-0.3,-0.6], at), p2=np.array([0.0,0.3,-0.3], at), p3=np.array([-0.3,-0.3,0.0], at), dcolor=np.array([1.0,0.0,1.0], at), scolor=np.array([1.0,0.1,1.0], at), kd=0.1, ks=0.8, ka=0.1, kgls=32.0, refl=0.0, light=light)
    # yellowTriangle = Triangle(p1=np.array([-0.2,0.1,0.1], at), p2=np.array([-0.2,-0.5,0.2], at), p3=np.array([-0.2,0.1,-0.3], at), dcolor=np.array([1.0,1.0,0.0], at), scolor=np.array([1.0,1.0,0.1], at), kd=0.2, ks=0.7, ka=0.1, kgls=4.0, refl=0.0, light=light)
    reflectiveTriangle = Triangle(p1=np.array([-0.2,0.1,0.1], at), p2=np.array([-0.2,-0.5,0.2], at), p3=np.array([-0.2,0.1,-0.3], at), dcolor=np.array([1.0,1.0,1.0], at), scolor=np.array([1.0,1.0,1.0], at), kd=0.0, ks=0.7, ka=0.1, kgls=40.0, refl=0.9, light=light)
    
    # rayTracer = RayTracer(camera, light, [blueSphere, redSphere, greenSphere, purpleTriangle, yellowTriangle], width, height)
    rayTracer = RayTracer(camera, light, [blueSphere, redSphere, greenSphere, purpleTriangle, reflectiveTriangle], width, height)
    rayTracer.traceToPPM("CustomScene.3")

def main():
    width = 1000
    height = 1000
    # width = 512
    # height = 512

    # tracePurpleSphere(width, height)
    # traceMultiSpheres(width, height)
    # traceScene1(width, height)
    # traceScene2(width, height)
    traceCustomScene(width, height)

if __name__ == "__main__":
    main()