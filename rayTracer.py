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

class Point():
    def __init__(self, point, index, dist):
        self.point = point
        self.index = index
        self.dist = dist

    def getDist(self): return self.dist

class Ray():
    def __init__(self, origin=np.array([0,0,0]), direction=np.array([1,1,1])):
        self.origin = np.array([origin[0], origin[1], origin[2]])
        self.direction = np.array([direction[0], direction[1], direction[2]])
        self.direction = self.direction/np.linalg.norm(self.direction)
        self.reflectionRay = None
        self.shadowRay = None

    def castReflectionRay(self, point, direction):
        self.reflectionRay = Ray(point, direction)
        return self.reflectionRay
    
    def castShadowRay(self, point, direction):
        self.shadowRay = Ray(point, direction)
        return self.shadowRay

class DirectionalLight():
    def __init__(self, dirToLight = np.array([0.0,1.0,0.0]), lightColor = np.array([1,1,1]), aLight = np.array([0.0,0.0,0.0]), bgColor = np.array([0.2,0.2,0.2])):
        self.dirToLight = dirToLight
        self.lightColor = lightColor
        self.aLight = aLight
        self.bgColor = bgColor


class Camera():
    def __init__(self, lookAt=np.array([0,0,0]), lookFrom=np.array([0,0,1]), lookUp=np.array([0,1,0]), width=1920, height=1080, fov=90):
        self.lookAt = lookAt
        self.lookFrom = lookFrom
        self.lookUp = lookUp/np.linalg.norm(lookUp)
        self.width = width
        self.height = height
        self.fov = fov

        ar = width/height
        mag = np.linalg.norm(lookAt - lookFrom)
        tanx = np.tan(np.radians(self.fov/2))
        tany = np.tan(np.radians(self.fov/ar/2))
        self.xext = abs(tanx * mag)
        self.yext = abs(tany * mag)

        self.hpx = self.xext / width
        self.hpy = self.yext / height

    def makeRay(self, i, j):
        u = -self.xext + self.hpx * (2 * i + 1)
        v = self.yext - self.hpy * (2 * j + 1)
        w = self.lookAt[2]
        origin = self.lookFrom
        direction = np.array([u, v, w]) - origin
        return Ray(origin, direction)

class Object():
    def __init__(self, dcolor = np.array([1,1,1]), scolor = np.array([1,1,1]), kd = .2, ks = .6, ka = .2, kgls = 2, refl = 0, light = DirectionalLight()):
        self.kd = kd
        self.ks = ks
        self.ka = ka
        self.kgls = kgls
        self.refl = refl
        self.dcolor = dcolor
        self.scolor = scolor
        self.light = light

    def getReflectv(self, normal):
        prod = np.dot(normal, self.light.dirToLight)
        reflectv = 2 * normal * prod - self.light.dirToLight
        return reflectv / np.linalg.norm(reflectv)

    def phongShading(self, normal, lookDir, shadow):
        ambient = self.light.aLight * self.ka * self.dcolor

        if not shadow:
            prod = np.dot(normal, self.light.dirToLight)
            diffuse = self.kd * self.light.lightColor * self.dcolor * max(0.0, prod)

            if self.refl == 0:
                reflectv = 2 * normal * prod - self.light.dirToLight
                reflectv /= np.linalg.norm(reflectv)
                specular = self.ks * self.light.lightColor * self.scolor * max(0.0, np.dot(lookDir, reflectv))**self.kgls
            else: specular = np.array([0,0,0])

            light_total = ambient + specular + diffuse
        else:
            light_total = ambient

        return np.clip(light_total*255, 0, 255)
    
    def intersectRay(self, ray: Ray, doubleSided: bool = False): 
        pass

class Sphere(Object):
    def __init__(self, center = np.array([0,0,0]), radius = 1, dcolor = np.array([1,1,1]), scolor = np.array([1,1,1]), kd = .3, ks = .6, ka = .1, kgls = 0, refl = 0, light = DirectionalLight()):
        super().__init__(dcolor=dcolor, scolor=scolor, kd=kd, ks=ks, ka=ka, kgls=kgls, refl=refl, light=light)
        self.center = center
        self.radius = radius
        self.normal = None

    def getColor(self, point, lookDir, shadow):
        self.normal = (point - self.center) / self.radius
        return self.phongShading(self.normal, lookDir, shadow)

    def intersectRay(self, ray: Ray, doubleSided = False):
        oc = self.center - ray.origin
        inside = np.linalg.norm(oc) < self.radius
        tca = ray.direction.dot(oc)
        if not inside and tca < 0: return None
        thcsquare = np.square(self.radius) - oc.dot(oc) + np.square(tca)
        if thcsquare < 0: return None

        if (inside): t = tca + np.sqrt(thcsquare)
        else: t = tca - np.sqrt(thcsquare)

        return ray.origin + ray.direction * t

class Triangle(Object):
    def __init__(self, p1=np.array([1,0,0]), p2=np.array([0,1,0]), p3=np.array([0,0,1]), dcolor=np.array([1, 1, 1]), scolor=np.array([1, 1, 1]), kd=0.2, ks=0.6, ka=0.2, kgls=2, refl = 0, light=DirectionalLight()):
        super().__init__(dcolor, scolor, kd, ks, ka, kgls, refl, light)
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        # v1 = self.p1 - self.p2
        # v2 = self.p1 - self.p3

        v1 = self.p3 - self.p2
        v2 = self.p1 - self.p2

        n = np.cross(v1, v2)
        self.normal = n / np.linalg.norm(n)

        self.d = -(p1[0] * self.normal[0] + p1[1] * self.normal[1] + p1[2] * self.normal[2])

    def getColor(self, point, lookDir, shadow):
        return self.phongShading(self.normal, lookDir, shadow)

    def intersectRay(self, ray: Ray, doubleSided = False):
        vd = np.dot(self.normal, ray.direction)
        if vd == 0: return None
        if (not doubleSided) and vd > 0: return None
        vo = -(self.normal.dot(ray.origin) + self.d)
        t = vo / vd
        if t < 0: return None
        
        p = ray.origin + ray.direction * t

        if vd > 0 and doubleSided:
            self.normal *= -1
            intersect = np.dot(-self.normal, np.cross((self.p2 - self.p1), (p - self.p1))) > 0 and np.dot(-self.normal, np.cross((self.p3 - self.p2), (p - self.p2))) > 0 and np.dot(-self.normal, np.cross((self.p1 - self.p3), (p - self.p3))) > 0
            self.normal *= -1
        else:
            intersect = np.dot(self.normal, np.cross((self.p2 - self.p1), (p - self.p1))) > 0 and np.dot(self.normal, np.cross((self.p3 - self.p2), (p - self.p2))) > 0 and np.dot(self.normal, np.cross((self.p1 - self.p3), (p - self.p3))) > 0

        return p if intersect else None


class RayTracer():
    def __init__(self, camera:Camera, light:DirectionalLight, objects:list[Object]=[], width:int=512, height:int=512):
        self.camera = camera
        self.objects = objects
        self.light = light
        self.width = width
        self.height = height

    def castShadowRay(self, ray, ignored = -1):
        for i in range(len(self.objects)):
            if i == ignored: continue
            point = self.objects[i].intersectRay(ray, True)
            if point is not None: return True
        return False

    def castRay(self, ray, lookDir, depth = 0, ignored = -1):
        points:list[Point] = []
        for i in range(len(self.objects)):
            if i == ignored: continue
            point = self.objects[i].intersectRay(ray)
            if point is None: continue
            dist = np.linalg.norm(point - ray.origin)
            points.append(Point(point, i, dist))
        if len(points) == 0:
            c = np.clip(self.light.bgColor*255, 0, 255)
        else:
            p = sorted(points, key=Point.getDist)[0]
            shadow = self.castShadowRay(Ray(p.point, self.light.dirToLight), p.index)
            c = self.objects[p.index].getColor(p.point, lookDir, shadow)

            if (depth < maxDepth) and (self.objects[p.index].refl > 0):
                m = self.castRay(Ray(p.point, self.objects[p.index].normal), lookDir, depth + 1, p.index) * self.objects[p.index].refl
                if m is not None: c = m + (1 - self.objects[p.index].refl) * c
        return np.clip(c.astype(int), 0, 255)

    def traceToPPM(self, imageName):
        lookDir = self.camera.lookFrom - self.camera.lookAt
        lookDir /= np.linalg.norm(lookDir)
        with open(f"{imageName}.ppm", 'wb') as f:
            f.write(f'P3\n{self.width} {self.height}\n255\n'.encode())
            for h in range(self.height):
                for w in range(self.width):
                    ray = self.camera.makeRay(w, h)
                    c = self.castRay(ray, lookDir)
                    f.write(f"{c[0]} {c[1]} {c[2]}  ".encode())
                f.write("\n".encode())
