import numpy as np

maxDepth = 2

def makeXRotate(rot):
	rad = np.radians(rot)
	cos = np.cos(rad)
	sin = np.sin(rad)
	return np.array([[1,0,0,0],[0,cos,-sin,0],[0,sin,cos,0],[0,0,0,1]])
def makeYRotate(rot): 
	rad = np.radians(rot)
	cos = np.cos(rad)
	sin = np.sin(rad)
	return np.array([[cos,0,sin,0],[0,1,0,0],[-sin,0,cos,0],[0,0,0,1]])
def makeZRotate(rot):
	rad = np.radians(rot)
	cos = np.cos(rad)
	sin = np.sin(rad)
	return np.array([[cos,-sin,0,0],[sin,cos,0,0],[0,0,1,0],[0,0,0,1]])

class PointThing():
    def __init__(self, point, index, dist):
        self.point = point
        self.index = index
        self.dist = dist

    def getDist(self): return self.dist

class Ray():
    def __init__(self, origin = np.array([0,0,0]), direction = np.array([1,1,1])):
        self.origin = origin
        self.direction = direction/np.linalg.norm(direction)

class DirectionalLight():
    def __init__(self, dirToLight = np.array([0.0,1.0,0.0]), lightColor = np.array([1,1,1]), aLight = np.array([0.0,0.0,0.0]), bgColor = np.array([0.2,0.2,0.2])):
        self.dirToLight = dirToLight
        self.lightColor = lightColor
        self.aLight = aLight
        self.bgColor = bgColor

class Camera():
    def __init__(self, lookAt = np.array([0,0,0]), lookFrom = np.array([0,0,1]), lookUp = np.array([0,1,0]), width = 1920, height = 1080, fovw = 90, fovh = 90):
        self.lookAt = lookAt
        self.lookFrom = lookFrom
        self.lookUp = lookUp
        self.width = width
        self.height = height
        self.fovx = fovw
        self.fovy = fovh

        # self.ar = width / height

        self.xext = abs(np.tan(np.radians(self.fovx/2)) * np.linalg.norm(lookAt - lookFrom))
        self.yext = abs(np.tan(np.radians(self.fovy/2)) * np.linalg.norm(lookAt - lookFrom) / (width / height))
        # self.xext = np.round(self.xext)
        # self.yext = np.round(self.yext)

        self.hpx = 2 * self.xext / width
        self.hpy = 2 * self.yext / height

    def makeRay(self, w, h):
        x = -self.xext + self.hpx * (w + 0.5)
        y = self.yext - self.hpy * (h + 0.5)

        return Ray(origin=self.lookFrom, direction=np.array([x, y, self.lookAt[2]]) - self.lookFrom)
        ray = Ray(origin=self.lookFrom, direction=np.array([x, y, self.lookAt[2]]) - self.lookFrom)
        return ray

class Object():
    def __init__(self, dcolor = np.array([1,1,1]), scolor = np.array([1,1,1]), kd = .2, ks = .6, ka = .2, kgls = 2, light = DirectionalLight()):
        self.kd = kd
        self.ks = ks
        self.ka = ka
        self.kgls = kgls
        self.dcolor = dcolor
        self.scolor = scolor
        self.light = light

    def phongShading(self, normal, lookDir):
        ambient = self.light.aLight * self.ka * self.dcolor

        prod = np.dot(normal, self.light.dirToLight)
        diffuse = self.kd * self.light.lightColor * self.dcolor * max(0.0, prod)

        reflectv = 2 * normal * prod - self.light.dirToLight
        reflectv /= np.linalg.norm(reflectv)
        specular = self.ks * self.light.lightColor * self.scolor * max(0.0, np.dot(lookDir, reflectv))**self.kgls
        
        light_total = ambient + specular + diffuse

        return np.clip(light_total*255, 0, 255).astype(int)
    
    def intersectRay(self, ray: Ray): pass

class Sphere(Object):
    def __init__(self, center = np.array([0,0,0]), radius = 1, dcolor = np.array([1,1,1]), scolor = np.array([1,1,1]), kd = .3, ks = .6, ka = .1, kgls = 0, light = DirectionalLight()):
        super().__init__(dcolor=dcolor, scolor=scolor, kd=kd, ks=ks, ka=ka, kgls=kgls, light=light)
        self.center = center
        self.radius = radius

    def getColor(self, point, lookDir):
        normal = (point - self.center) / self.radius
        return self.phongShading(normal, lookDir)

    def intersectRay(self, ray: Ray):
        oc = self.center - ray.origin
        inside = np.linalg.norm(oc) < self.radius
        tca = ray.direction.dot(oc)
        if not inside and tca < 0: return None
        thcsquare = np.square(self.radius) - oc.dot(oc) + np.square(tca)
        if thcsquare < 0: return None

        if (inside): t = tca + np.sqrt(thcsquare)
        else: t = tca - np.sqrt(thcsquare)

        return ray.origin + ray.direction * t

class RayTracer():
    def __init__(self, objects:list[Object]=[], camera:Camera=Camera(), light:DirectionalLight=DirectionalLight(), width:int=512, height:int=512):
        self.camera = camera
        self.objects = objects
        self.light = light
        self.width = width
        self.height = height
    
    def traceToPPM(self, imageName):
        lookDir = self.camera.lookFrom - self.camera.lookAt
        lookDir /= np.linalg.norm(lookDir)
        with open(f"{imageName}.ppm", 'wb') as f:
            f.write(f'P3\n{self.width} {self.height}\n255\n'.encode())
            for h in range(self.height):
                for w in range(self.width):
                    ray = self.camera.makeRay(w, h)
                    points = []
                    for i in range(len(self.objects)):
                        point = self.objects[i].intersectRay(ray)
                        if point is None: continue
                        dist = np.linalg.norm(point - ray.origin)
                        points.append(PointThing(point, i, dist))
                    if len(points) == 0:
                        c = np.clip(self.light.bgColor*255, 0, 255).astype(int)
                    else:
                        pointSort = sorted(points, key=PointThing.getDist)
                        p = pointSort[0]
                        c = self.objects[p.index].getColor(p.point, lookDir)
                    f.write(f"{c[0]} {c[1]} {c[2]}  ".encode())
                f.write("\n".encode())

def main():
    at = np.float32

    dirToLight = np.array([1.0,1.0,1.0], at)
    lightColor = np.array([1.0,1.0,1.0], at)
    aLight = np.array([0.1,0.1,0.1], at)
    bgColor = np.array([0.2,0.2,0.2], at)
    light = DirectionalLight(dirToLight=dirToLight, lightColor=lightColor, aLight=aLight, bgColor=bgColor)

    # width = 1920
    # height = 1080
    width = 800
    height = 800
    lookAt=np.array([0.0,0.0,0.0], at)
    lookFrom=np.array([0.0,0.0,1.0], at)
    lookUp=np.array([0.0,1.0,0.0], at)
    camera = Camera(width=width, height=height, lookAt=lookAt, lookFrom=lookFrom, lookUp=lookUp, fovh=90, fovw=90)

    # purpleSphere = Sphere(center=np.array([0.0,0.0,0.0], at), dcolor=np.array([1.0,0.0,1.0], at), scolor=np.array([1.0,1.0,1.0], at), radius=0.4, kd=0.7, ks=0.2, ka=0.1, kgls=16.0, light=light)
    # rayTracer = RayTracer([purpleSphere], camera, light, width, height)
    # rayTracer.traceToPPM("purpleSphereImageTest")

    whiteSphere = Sphere(center=np.array([0.45,0.0,-0.15], at), dcolor=np.array([1.0,1.0,1.0], at), scolor=np.array([1.0,1.0,1.0], at), radius=0.15, kd=0.8, ks=0.1, ka=0.3, kgls=4.0, light=light)
    whiteSphere = Sphere(center=np.array([0.45,0.0,-0.15], at), dcolor=np.array([1.0,1.0,1.0], at), scolor=np.array([1.0,1.0,1.0], at), radius=0.15, kd=0.6, ks=0.1, ka=0.3, kgls=4.0, light=light)
    redSphere = Sphere(center=np.array([0.0,0.0,-0.1], at), dcolor=np.array([1.0,0.0,0.0], at), scolor=np.array([1.0,1.0,1.0], at), radius=0.2, kd=0.6, ks=0.3, ka=0.1, kgls=32.0, light=light)
    greenSphere = Sphere(center=np.array([-0.6,0.0,0.0], at), dcolor=np.array([0.0,1.0,0.0], at), scolor=np.array([0.5,1.0,0.5], at), radius=0.3, kd=0.7, ks=0.2, ka=0.1, kgls=64.0, light=light)
    blueSphere = Sphere(center=np.array([0.0,-10000.5,0.0], at), dcolor=np.array([0.0,0.0,1.0], at), scolor=np.array([1.0,1.0,1.0], at), radius=10000.0, kd=0.9, ks=0.0, ka=0.1, kgls=16.0, light=light)
    rayTracer = RayTracer([whiteSphere, redSphere, greenSphere, blueSphere], camera, light, width, height)
    
    rayTracer.traceToPPM("multipleSpheresImage")

if __name__ == "__main__":
    main()