
import re
import math
import FreeCAD
import Part
import Draft
import DraftVecUtils
from FreeCAD import Vector
from draftutils.messages import _err, _msg, _wrn

def arcend2center(lastvec, currentvec, rx, ry,
                  xrotation=0.0, correction=False):
    '''Calculate the possible centers for an arc in endpoint parameterization.

    Calculate (positive and negative) possible centers for an arc given in
    ``endpoint parametrization``.
    See http://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes

    the sweepflag is interpreted as: sweepflag <==>  arc is travelled clockwise

    Parameters
    ----------
    lastvec : Base::Vector3
        First point of the arc.
    currentvec : Base::Vector3
        End point (current) of the arc.
    rx : float
        Radius of the ellipse, semi-major axis in the X direction.
    ry : float
        Radius of the ellipse, semi-minor axis in the Y direction.
    xrotation : float, optional
        Default is 0. Rotation around the Z axis, in radians (CCW).
    correction : bool, optional
        Default is `False`. If it is `True`, the radii will be scaled
        by a factor.

    Returns
    -------
    list, (float, float)
        A tuple that consists of one list, and a tuple of radii.
    [(positive), (negative)], (rx, ry)
        The first element of the list is the positive tuple,
        the second is the negative tuple.
    [(Base::Vector3, float, float),
    (Base::Vector3, float, float)], (float, float)
        Types
    [(vcenter+, angle1+, angledelta+),
    (vcenter-, angle1-, angledelta-)], (rx, ry)
        The first element of the list is the positive tuple,
        consisting of center, angle, and angle increment;
        the second element is the negative tuple.
    '''
    # scalefacsign = 1 if (largeflag != sweepflag) else -1
    rx = float(rx)
    ry = float(ry)
    v0 = lastvec.sub(currentvec)
    v0.multiply(0.5)
    m1 = FreeCAD.Matrix()
    m1.rotateZ(-xrotation)  # eq. 5.1
    v1 = m1.multiply(v0)
    if correction:
        eparam = v1.x**2 / rx**2 + v1.y**2 / ry**2
        if eparam > 1:
            eproot = math.sqrt(eparam)
            rx = eproot * rx
            ry = eproot * ry
    denom = rx**2 * v1.y**2 + ry**2 * v1.x**2
    numer = rx**2 * ry**2 - denom
    results = []

    # If the division is very small, set the scaling factor to zero,
    # otherwise try to calculate it by taking the square root
    if abs(numer/denom) < 10**(-Draft.precisionSVG()):
        scalefacpos = 0
    else:
        try:
            scalefacpos = math.sqrt(numer/denom)
        except ValueError:
            _msg("sqrt({0}/{1})".format(numer, denom))
            scalefacpos = 0

    # Calculate two values because the square root may be positive or negative
    for scalefacsign in (1, -1):
        scalefac = scalefacpos * scalefacsign
        # Step2 eq. 5.2
        vcx1 = Vector(v1.y * rx/ry, -v1.x * ry/rx, 0).multiply(scalefac)
        m2 = FreeCAD.Matrix()
        m2.rotateZ(xrotation)
        centeroff = currentvec.add(lastvec)
        centeroff.multiply(0.5)
        vcenter = m2.multiply(vcx1).add(centeroff)  # Step3 eq. 5.3
        # angle1 = Vector(1, 0, 0).getAngle(Vector((v1.x - vcx1.x)/rx,
        #                                          (v1.y - vcx1.y)/ry,
        #                                          0))  # eq. 5.5
        # angledelta = Vector((v1.x - vcx1.x)/rx,
        #                     (v1.y - vcx1.y)/ry,
        #                     0).getAngle(Vector((-v1.x - vcx1.x)/rx,
        #                                        (-v1.y - vcx1.y)/ry,
        #                                        0))  # eq. 5.6
        # we need the right sign for the angle
        angle1 = DraftVecUtils.angle(Vector(1, 0, 0),
                                     Vector((v1.x - vcx1.x)/rx,
                                            (v1.y - vcx1.y)/ry,
                                            0))  # eq. 5.5
        angledelta = DraftVecUtils.angle(Vector((v1.x - vcx1.x)/rx,
                                                (v1.y - vcx1.y)/ry,
                                                0),
                                         Vector((-v1.x - vcx1.x)/rx,
                                                (-v1.y - vcx1.y)/ry,
                                                0))  # eq. 5.6
        results.append((vcenter, angle1, angledelta))

        if rx < 0 or ry < 0:
            _wrn("Warning: 'rx' or 'ry' is negative, check the SVG file")

    return results, (rx, ry)


def makewire(path, checkclosed=False, donttry=False):
    '''Try to make a wire out of the list of edges.

    If the wire functions fail or the wire is not closed,
    if required the TopoShapeCompoundPy::connectEdgesToWires()
    function is used.

    Parameters
    ----------
    path : Part.Edge
        A collection of edges
    checkclosed : bool, optional
        Default is `False`.
    donttry : bool, optional
        Default is `False`. If it's `True` it won't try to check
        for a closed path.

    Returns
    -------
    Part::Wire
        A wire created from the ordered edges.
    Part::Compound
        A compound made of the edges, but unable to form a wire.
    '''
    if not donttry:
        try:
            sh = Part.Wire(path)
            # sh = Part.Wire(path)
            isok = (not checkclosed) or sh.isClosed()
            if len(sh.Edges) != len(path):
                isok = False
        # BRep_API: command not done
        except Part.OCCError:
            isok = False
    if donttry or not isok:
        # Code from wmayer forum p15549 to fix the tolerance problem
        # original tolerance = 0.00001
        comp = Part.Compound(path)
        _sh = comp.connectEdgesToWires(False, 10**(-Draft.precisionSVG()))
        sh = _sh.Wires[0]
        if len(sh.Edges) != len(path):
            _wrn("Unable to form a wire")
            sh = comp
    return sh


def arccenter2end(center, rx, ry, angle1, angledelta, xrotation=0.0):
    '''Calculate start and end points, and flags of an arc.

    Calculate start and end points, and flags of an arc given in
    ``center parametrization``.
    See http://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes

    Parameters
    ----------
    center : Base::Vector3
        Coordinates of the center of the ellipse.
    rx : float
        Radius of the ellipse, semi-major axis in the X direction
    ry : float
        Radius of the ellipse, semi-minor axis in the Y direction
    angle1 : float
        Initial angle in radians
    angledelta : float
        Additional angle in radians
    xrotation : float, optional
        Default 0. Rotation around the Z axis

    Returns
    -------
    v1, v2, largerc, sweep
        Tuple indicating the end points of the arc, and two boolean values
        indicating whether the arc is less than 180 degrees or not,
        and whether the angledelta is negative.
    '''
    vr1 = Vector(rx * math.cos(angle1), ry * math.sin(angle1), 0)
    vr2 = Vector(rx * math.cos(angle1 + angledelta),
                 ry * math.sin(angle1 + angledelta),
                 0)
    mxrot = FreeCAD.Matrix()
    mxrot.rotateZ(xrotation)
    v1 = mxrot.multiply(vr1).add(center)
    v2 = mxrot.multiply(vr2).add(center)
    fa = ((abs(angledelta) / math.pi) % 2) > 1  # < 180 deg
    fs = angledelta < 0
    return v1, v2, fa, fs


class FaceTreeNode:
    '''Building Block of a tree structure holding one-closed-wire faces 
       sorted after heir enclosure of each other.
       This class only works with faces that have exactly one closed wire
    '''
    face     : Part.Face
    children : list
    name     : str

    
    def __init__(self, face=None, name="root"):
        super().__init__()
        self.face = face
        self.name = name
        self.children = [] 

      
    def insert (self, face, name):
        ''' takes a single-wire face, and inserts it into the tree 
            depending on its enclosure in/of in already added faces

            Parameters
            ----------
            face : Part.Face
                   single closed wire face to be added to the tree
            name : str
                   face identifier       
        ''' 
        if face.Area < 10 * (10**-Draft.precisionSVG())**2: 
            # drop the plane, it's less than 10 resolution units in size
            return 
        for node in self.children:
            if  node.face.Area > face.Area:
                # new face could be encompassed
                if (face.distToShape(node.face)[0] == 0.0 and 
                    face.Wires[0].distToShape(node.face.Wires[0])[0] != 0.0):
                    # it is encompassed - enter next tree layer
                    node.insert(face, name)
                    return
            else:
                # new face could encompass
                if (node.face.distToShape(face)[0] == 0.0 and
                    node.face.Wires[0].distToShape(face.Wires[0])[0] != 0.0):
                    # it does encompass the current child nodes face
                    # create new node from face
                    new = FaceTreeNode(face, name)
                    # swap the new one with the child node 
                    self.children.remove(node)
                    self.children.append(new)
                    # add former child node as child to the new node
                    new.children.append(node)
                    return
        # the face is not encompassing and is not encompassed (from) any
        # other face, we add it as new child 
        new = FaceTreeNode(face, name)
        self.children.append(new)

     
    def makeCuts(self):
        ''' recursively traverse the tree and cuts all faces in even 
            numbered tree levels with their direct childrens faces. 
            Additionally the tree is shrunk by removing the odd numbered 
            tree levels.                 
        '''
        result = self.face
        if not result:
            for node in self.children:
                node.makeCuts()
        else:
            new_children = []
            for node in self.children:
                result = result.cut(node.face)
                for subnode in node.children:
                    subnode.makeCuts()
                    new_children.append(subnode)
            self.children = new_children
            self.face = result

                
    def traverse(self, function):
        function(self)
        for node in self.children:
            node.traverse(function) 

           
    def flatten(self):
        ''' creates a flattened list of face-name tuples from the facetree
            content
        '''
        result = []
        result.append((self.name, self.face))
        for node in self.children:
            result.extend(node.flatten())
        return result  
        
  
  
class SvgPath_Element:
    ''' Data class that holds the raw information of a single svg path edge.
	'''
    vertexes : list[Vector]
    values   : list[float]
    type     : str
    
    def __init__(self, ptype):
        super().__init__()
        """initialize."""
        self.type = ptype 
        self.vertexes = []
        self.values = []
        

class SvgPath:
    """Parse SVG path data and create FreeCAD Shapes."""

    commands : list[tuple]
    pointsre : re.Pattern
    data     : dict
    paths    : list[list[SvgPath_Element]]
    shapes   : list[list[Part.Shape]] 
    faces    : FaceTreeNode  
    name     : str

    def __init__(self, data, name):
        super().__init__()
        """Evaluate path data and initialize."""
        _op = '([mMlLhHvVaAcCqQsStTzZ])'
        _op2 = '([^mMlLhHvVaAcCqQsStTzZ]*)'
        _command = '\\s*?' + _op + '\\s*?' + _op2 + '\\s*?'
        pathcommandsre = re.compile(_command, re.DOTALL)
    
        _num = '[-+]?[0-9]*\\.?[0-9]+'
        _exp = '([eE][-+]?[0-9]+)?'
        _point = '(' + _num + _exp + ')'
        self.commands = pathcommandsre.findall(' '.join(data['d']))
        self.pointsre = re.compile(_point, re.DOTALL)
        self.data = data
        self.paths = []
        self.shapes = []
        self.faces = None
        self.name = name
        
    def __finalizePath(self, path):
        ''' This is kind of a finalizing function for the lists of raw
            SvgPath_Elements derived by calling 'parse'.

            It compares start and end points of the submitted path and 
            replaces them with the center between them if they are the 
            same in respect to the configured precision, but still 
            differ mathematically. 
            the finalized path is then added to the pool of paths
            
            Parameters
            ----------
            path : list[SvgPath_Element]
                   a list of SvgPath_Elements representing a single, 
                   possibly closed wire.   
        '''
        # are start and endpoint nearly but not completely the same?
        if DraftVecUtils.equals(path[0].vertexes[0], 
                                path[-1].vertexes[-1], Draft.precisionSVG()) \
           and (   (path[0].vertexes[0].x != path[-1].vertexes[-1].x) \
                or (path[0].vertexes[0].y != path[-1].vertexes[-1].y) ):
            # calculate center between start and endpoint
            dx = (path[0].vertexes[0].x + path[-1].vertexes[1].x) / 2.0
            dy = (path[0].vertexes[0].y + path[-1].vertexes[1].y) / 2.0

            path[ 0].vertexes[ 0] = Vector(dx, dy) # replace startpoint
            path[-1].vertexes[-1] = Vector(dx, dy) # replace endpoint
        self.paths.append(path)

    
    def parse(self):
        ''' This function creates lists of SvgPath_Elements from raw svg path 
            data. It's supposed to be called direct after SvgPath Object
            creation.
            Afterwards the list of SvgPath_Elements can be refined before the 
            function 'createShapes' produces a series of Shapes.
        '''
        path = []
        self.paths = []
        currentvec = Vector(0,0,0)
        startvec = currentvec
        for d, pointsstr in self.commands:
            relative = d.islower()
            _points = self.pointsre.findall(pointsstr.replace(',', ' '))
            pointlist = [float(number) for number, exponent in _points]

            if (d == "M" or d == "m"):
                if path:
                    self.__finalizePath(path)
                    path = []

                x = pointlist.pop(0)
                y = pointlist.pop(0)
                if relative:
                    startvec = currentvec = startvec.add(Vector(x, -y, 0))
                else:
                    startvec = currentvec = Vector(x, -y, 0)
                self.lastpole = None
            if (       (d == 'M' or d == 'm') and pointlist) \
                    or (d == "L" or d == "l") \
                    or (d == "H" or d == "h") \
                    or (d == "V" or d == "v"):
                x = 0
                y = 0
                while pointlist:
                    if not (d == "V" or d == "v"):
                        x = pointlist.pop(0)
                    else:
                        if relative:
                            x = 0
                        else:
                            x = currentvec.x
                    if not (d == "H" or d == "h"):
                        y = pointlist.pop(0)
                    else:
                        if relative:
                            y = 0
                        else:
                            y = -currentvec.y
                    ele = SvgPath_Element("Line")
                    ele.vertexes.append(currentvec)
                    if relative:
                        currentvec = currentvec.add(Vector(x, -y, 0))
                    else:
                        currentvec = Vector(x, -y, 0)
                        
                    if not DraftVecUtils.equals(ele.vertexes[0], currentvec, Draft.precisionSVG()):
                        ele.vertexes.append(currentvec)
                        path.append(ele)
                    else:
                        currentvec = ele.vertexes[0]
            elif (d == "A" or d == "a"):
                piter = zip(pointlist[0::7], pointlist[1::7],
                            pointlist[2::7], pointlist[3::7],
                            pointlist[4::7], pointlist[5::7],
                            pointlist[6::7])
                for (rx, ry, xrotation,  largeflag, sweepflag, x, y) in piter:
                    # support for large-arc and x-rotation is missing
                    ele = SvgPath_Element("Arc")
                    ele.vertexes.append(currentvec)
                    if relative:
                        currentvec = currentvec.add(Vector(x, -y, 0))
                    else:
                        currentvec = Vector(x, -y, 0)
                    if (DraftVecUtils.equals(ele.vertexes[0], currentvec, Draft.precisionSVG())):
                        currentvec = ele.vertexes[0]
                        continue
                    ele.vertexes.append(currentvec)
                    ele.values.append(rx)
                    ele.values.append(ry)
                    ele.values.append(xrotation)
                    ele.values.append(largeflag)
                    ele.values.append(sweepflag)
                    ele.values.append(x)
                    ele.values.append(y)
                    path.append(ele)
            elif (d == "C" or d == "c") or (d == "S" or d == "s"):
                smooth = (d == 'S' or d == 's')
                if smooth:
                    piter = list(zip(pointlist[2::4],
                                     pointlist[3::4],
                                     pointlist[0::4],
                                     pointlist[1::4],
                                     pointlist[2::4],
                                     pointlist[3::4]))
                else:
                    piter = list(zip(pointlist[0::6],
                                     pointlist[1::6],
                                     pointlist[2::6],
                                     pointlist[3::6],
                                     pointlist[4::6],
                                     pointlist[5::6]))
                for p1x, p1y, p2x, p2y, x, y in piter:
                    ele = SvgPath_Element("CubicBezier")
                    ele.vertexes.append(currentvec)
                    if relative:
                        currentvec = currentvec.add(Vector(x, -y, 0))
                    else:
                        currentvec = Vector(x, -y, 0)
                    if DraftVecUtils.equals(ele.vertexes[0], currentvec, Draft.precisionSVG()):
                        currentvec = ele.vertexes[0]
                        continue
                    ele.vertexes.append(currentvec)
                    ele.values.append(smooth)    
                    ele.values.append(relative)    
                    ele.values.append(p1x)    
                    ele.values.append(p1y)    
                    ele.values.append(p2x)    
                    ele.values.append(p2y)    
                    ele.values.append(x)    
                    ele.values.append(y)  
                    path.append(ele)
            elif (d == "Q" or d == "q") or (d == "T" or d == "t"):
                smooth = (d == 'T' or d == 't')
                if smooth:
                    piter = list(zip(pointlist[1::2],
                                     pointlist[1::2],
                                     pointlist[0::2],
                                     pointlist[1::2]))
                else:
                    piter = list(zip(pointlist[0::4],
                                     pointlist[1::4],
                                     pointlist[2::4],
                                     pointlist[3::4]))
                for px, py, x, y in piter:
                    ele = SvgPath_Element("QuadBezier")
                    ele.vertexes.append(currentvec)
                    if relative:
                        currentvec = currentvec.add(Vector(x, -y, 0))
                    else:
                        currentvec = Vector(x, -y, 0)
                    if DraftVecUtils.equals(ele.vertexes[0], currentvec, Draft.precisionSVG()):
                        currentvec = ele.vertexes[0]
                        continue
                    ele.vertexes.append(currentvec)
                    ele.values.append(smooth)    
                    ele.values.append(relative)    
                    ele.values.append(px)    
                    ele.values.append(py)    
                    ele.values.append(x)    
                    ele.values.append(y)  
                    path.append(ele)          
            elif (d == "Z") or (d == "z"):
                if len(path) < 2:
                    # this is no wire, possibly not even a vertex.
                    path = []
                    currentvec = startvec
                    continue
                
                if not DraftVecUtils.equals(path[0].vertexes[0], path[-1].vertexes[-1], Draft.precisionSVG()):
                    ele = SvgPath_Element("Line")
                    ele.vertexes.append(path[-1].vertexes[-1])
                    ele.vertexes.append(path[0].vertexes[0])
                    path.append(ele)
                self.__finalizePath(path)
                path = []
                    
        if path:
            self.__finalizePath(path)


    def createShapes(self):
        ''' This function generates lists of shapes from SvgPath_Element 
            lists.
            It's supposed to be called after the function 'parse', that
            generated SvgPath_Element lists from raw svg path data.
        '''
        self.shapes = []
        for path in self.paths:
            shape = []
            lastpole = None
            for ele in path:
                if (ele.type == "Line"):
                    seg = Part.LineSegment(ele.vertexes[0], ele.vertexes[-1]).toShape()
                elif (ele.type == "Arc"):
                    rx = ele.values[0]
                    ry = ele.values[1]
                    xrotation = ele.values[2]
                    largeflag = ele.values[3]
                    sweepflag = ele.values[4]
                    currentvec = ele.vertexes[-1]
                    lastvec = ele.vertexes[0]
                    chord = currentvec.sub(lastvec)
                    _precision = 10**(-Draft.precisionSVG())
                    # small circular arc
                    if (not largeflag) and abs(rx - ry) < _precision:
                        # perp = chord.cross(Vector(0, 0, -1))
                        # here is a better way to find the perpendicular
                        if sweepflag != 0:
                            # clockwise
                            perp = DraftVecUtils.rotate2D(chord,
                                                          -math.pi/2)
                        else:
                            # anticlockwise
                            perp = DraftVecUtils.rotate2D(chord, math.pi/2)
                        chord.multiply(0.5)
                        if chord.Length > rx:
                            a = 0
                        else:
                            a = math.sqrt(rx**2 - chord.Length**2)
                        s = rx - a
                        perp.multiply(s/perp.Length)
                        midpoint = lastvec.add(chord.add(perp))
                        _seg = Part.Arc(lastvec, midpoint, currentvec)
                        seg = _seg.toShape()
                    # big arc or elliptical arc
                    else:
                        # Calculate the possible centers for an arc
                        # in 'endpoint parameterization'.
                        _xrot = math.radians(-xrotation)
                        (solution, (rx, ry)) = arcend2center(lastvec, currentvec, 
                                                             rx, ry, 
                                                             xrotation=_xrot, 
                                                             correction=True)
                        # Chose one of the two solutions
                        negsol = (largeflag != sweepflag)
                        vcenter, angle1, angledelta = solution[negsol]
                        # print(angle1)
                        # print(angledelta)
                        if ry > rx:
                            rx, ry = ry, rx
                            swapaxis = True
                        else:
                            swapaxis = False
                        # print('Elliptical arc %s rx=%f ry=%f'
                        #       % (vcenter, rx, ry))
                        e1 = Part.Ellipse(vcenter, rx, ry)
                        if sweepflag:
                            # Step4
                            # angledelta = -(-angledelta % (2*math.pi))
                            # angledelta = (-angledelta % (2*math.pi))
                            angle1 = angle1 + angledelta
                            angledelta = -angledelta
                            # angle1 = math.pi - angle1
    
                        d90 = math.radians(90)
                        e1a = Part.Arc(e1,
                                       angle1 - swapaxis * d90,
                                       angle1 + angledelta
                                              - swapaxis * d90)
                        # e1a = Part.Arc(e1,
                        #                angle1 - 0 * swapaxis * d90,
                        #                angle1 + angledelta
                        #                       - 0 * swapaxis * d90)
                        seg = e1a.toShape()
                        if swapaxis:
                            seg.rotate(vcenter, Vector(0, 0, 1), 90)
                        _precision = 10**(-Draft.precisionSVG())
                        if abs(xrotation) > _precision:
                            seg.rotate(vcenter, Vector(0, 0, 1), -xrotation)
                        if sweepflag:
                            seg.reverse()
                            # DEBUG
                            # obj = self.doc.addObject("Part::Feature",
                            #                       'DEBUG %s' % pathname)
                            # obj.Shape = seg
                            # _seg = Part.LineSegment(lastvec, currentvec)
                            # seg = _seg.toShape()
                elif (ele.type == "CubicBezier"):
                    currentvec = ele.vertexes[-1]
                    lastvec    = ele.vertexes[ 0]
                    
                    smooth   = ele.values[0]
                    relative = ele.values[1]
                    p1x      = ele.values[2]
                    p1y      = ele.values[3]
                    p2x      = ele.values[4]
                    p2y      = ele.values[5]
    
                    if smooth:
                        if lastpole is not None and lastpole[0] == 'cubic':
                            pole1 = lastvec.sub(lastpole[1]).add(lastvec)
                        else:
                            pole1 = lastvec
                    else:
                        if relative:
                            pole1 = lastvec.add(Vector(p1x, -p1y, 0))
                        else:
                            pole1 = Vector(p1x, -p1y, 0)
                    if relative:
                        pole2 = lastvec.add(Vector(p2x, -p2y, 0))
                    else:
                        pole2 = Vector(p2x, -p2y, 0)
    
                    _precision = 10**(-Draft.precisionSVG())
                    _d1 = pole1.distanceToLine(lastvec, currentvec)
                    _d2 = pole2.distanceToLine(lastvec, currentvec)
                    if True and _d1 < _precision and _d2 < _precision:
                        seg = Part.LineSegment(lastvec, currentvec).toShape()
                    else:
                        b = Part.BezierCurve()
                        b.setPoles([lastvec, pole1, pole2, currentvec])
                        seg = b.toShape()
                    lastpole = ('cubic', pole2)                  
                              
                elif (ele.type == "QuadBezier"):
                    
                    currentvec = ele.vertexes[-1]
                    lastvec    = ele.vertexes[ 0]
                    
                    smooth   = ele.values[0]
                    relative = ele.values[1]
                    px       = ele.values[2]
                    py       = ele.values[3]
                    if smooth:
                        if (lastpole is not None \
                                and lastpole[0] == 'quadratic'):
                            pole = lastvec.sub(lastpole[1]).add(lastvec)
                        else:
                            pole = lastvec
                    else:
                        if relative:
                            pole = lastvec.add(Vector(px, -py, 0))
                        else:
                            pole = Vector(px, -py, 0)    
                    if not DraftVecUtils.equals(lastvec, currentvec, Draft.precisionSVG()):
                        _precision = 10**(-Draft.precisionSVG())
                        _distance = pole.distanceToLine(lastvec, currentvec)
                        if _distance < _precision:
                            seg = Part.LineSegment(lastvec, currentvec).toShape()
                        else:
                            b = Part.BezierCurve()
                            b.setPoles([lastvec, pole, currentvec])
                            seg = b.toShape()
                        lastvec = currentvec
                        lastpole = ('quadratic', pole)
                
                shape.append(seg)
            self.shapes.append(shape)

    def createFaces(self, fill=True):
        ''' This function tries to generate Faces from lists of shapes.
            If shapes form a closed wire and the fill Attribute is set, we 
            generate a closed Face. Otherwise we treat the shape as pure wire.
            
            Parameters
            ----------
            fill : Object/bool
                   if True or not None Faces are generated from closed shapes.
        '''
        cnt = 0;
        openShapes = []
        self.faces = FaceTreeNode()
        for sh in self.shapes:                               
            # The path should be closed by now
            # sh = makewire(path, True)
            add_wire = True
            sh = makewire(sh, checkclosed=True)
            
            if fill and len(sh.Wires) == 1 and sh.Wires[0].isClosed():
                try:
                    face = Part.Face(sh)
                    if  face.isValid():
                        add_wire = False
                        if not (face.Area < 10 * (10**-Draft.precisionSVG())**2):
                            self.faces.insert(face, self.name + "_f" + str(++cnt))
                        elif duh_ze_logs:
                            _msg("Drop face '{}' - too tiny. Area: {}".format(self.name + "_" + str(cnt), sh.Area))     
                except:
                    _msg("Failed to make a shape from path '{}'. This Path will be discarded.".format(self.name))
            if add_wire and sh.Length > (10**-Draft.precisionSVG()):
                openShapes.append((self.name + "_w" + str(++cnt), sh))
        self.shapes = openShapes


    def doCuts(self):
        ''' Exposes the FaceTreeNode.makeCuts function of the tree containing 
            closed wire faces.
            This function is called after creating closed Faces with
            'createFaces' in order to hollow faces encompassing others.
        '''      
        self.faces.makeCuts()


    def getShapeList(self):
        ''' Returns the resulting list of tuples containing name and face of 
            each created element.
        ''' 
        result = self.faces.flatten()
        result.extend(self.shapes)             
        return result
    
    
          
