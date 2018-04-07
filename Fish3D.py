from mayavi import mlab
from mayavi import tools as mayavitools
import numpy as np

#############################################################################################
#
#                                GEOMETRY FUNCTIONS
#
#############################################################################################

def project_vector_onto_plane(vector_to_project, plane_normal_vector):
    # Formula from http://www.maplesoft.com/support/help/Maple/view.aspx?path=MathApps%2FProjectionOfVectorOntoPlane
    u = vector_to_project
    n = plane_normal_vector
    return u - n * u.dot(n) / np.linalg.norm(n)**2

def matrix_to_move_fish_head_and_tail(modelhead, modeltail, modelvertical, newhead, newtail, newvertical, tol=1e-12):
    """Return a flattened 4D matrix to rotate, translate, and scale a 3-D fish model so its head 
    and tail points match those specified by 'newhead' and 'newtail' and the world vertical direction 'newvertical'"""
    modelvec = modeltail - modelhead
    modelvecnorm = np.linalg.norm(modelvec)
    unitmodelvec = modelvec / modelvecnorm
    newvec = newtail - newhead
    newvecnorm = np.linalg.norm(newvec)
    unitnewvec = newvec / newvecnorm
    # Putting together the individual transformations we compose at the end
    # Translate to the origin
    T1 = np.eye(4)
    T1[0:3,3] = -modelhead
    # Rescale to unit length
    S1 = (1/modelvecnorm) * np.eye(4)
    S1[3,3] = 1
    # Rotate the fish's body into the correct angle relative to the head (which is the temporary origin)
    # Math based on http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    v = np.cross(unitmodelvec,unitnewvec)
    s = np.linalg.norm(v)
    c = np.dot(unitmodelvec,unitnewvec)
    vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    R = np.eye(4) # Just use the identity matrix (no rotation) if the fish is parallel to the model, facing the same way
    if s != 0: # Use the main formula if the fish isn't parallel to the model
        R[0:3,0:3] = np.eye(3) + vx + vx.dot(vx)*(1-c)/(s*s) 
    else:
        if (unitmodelvec == -unitnewvec).all(): 
            # If the fish is parallel but facing the other way, reverse it.
            R[0:3,0:3] = -np.eye(3)
    # Build the matrix to turn the fish right-side up according to "newvertical" as the world's vertical direction. This means rotating the fish
    # about its axis, i.e. in the plane perpendicular to the fish's body axis. This requires projecting both the world vertical and rotated model vertical
    # into that plane and finding the angle between them, then rotating the fish by that angle.
    hr = R.dot(np.append(modelvertical,1));
    rotated_model_vertical = (hr/hr[3])[0:3]
    a1 = project_vector_onto_plane(rotated_model_vertical, unitnewvec)  # Project rotated model vertical into plane perpendicular to model axis
    a2 = project_vector_onto_plane(newvertical, unitnewvec)             # Project world vertical into plane perpendicular to model axis
    a1 = a1 / np.linalg.norm(a1)
    a2 = a2 / np.linalg.norm(a2)
    axis_rotation_angle = np.arccos(a1.dot(a2))  # Want to rotate the fish about its axis by the angle between a1 and a2
    # Adjust the sign of the angle as described at http://stackoverflow.com/questions/5188561/signed-angle-between-two-3d-vectors-with-same-origin-within-the-same-plane
    # axis_rotation_angle *= np.sign(unitnewvec.dot(np.cross(a1, a2)))
    # COMMENTING ABOVE LINE to turn my fish right-side-up... but does it mess up others? I'm not sure.
    t = 1 - np.cos(axis_rotation_angle)
    S = np.sin(axis_rotation_angle)
    C = np.cos(axis_rotation_angle)
    ux, uy, uz = unitnewvec
    R2 = np.eye(4) # Source of formula below for R2 to rotate about a specified axis at a specified angle: http://math.kennesaw.edu/~plaval/math4490/rotgen.pdf
    R2[0:3,0:3] = np.array([[t*ux*ux+C, t*ux*uy-S*uz, t*ux*uz+S*uy], [t*ux*uy+S*uz, t*uy*uy+C, t*uy*uz-S*ux], [t*ux*uz-S*uy, t*uy*uz+S*ux, t*uz*uz+C]])
    # Rescale to intended length
    S2 = newvecnorm * np.eye(4)
    S2[3,3] = 1
    # Translate to intended position
    T2 = np.eye(4)
    T2[0:3,3] = newhead
    # Put it all together into final matrix
    F = T2.dot(S2.dot(R2.dot(R.dot(S1.dot(T1)))))
    F[np.abs(F) < tol] = 0.0 # Trim any pointlessly tiny matrix elements to zero
    return F.flatten()

#############################################################################################
#
#                                FISH 3-D MODEL DATA
#
#############################################################################################

# For all three fish models:
# All fish were modeled based on an image of the fish, facing to the right, traced from "Left" viewpoint in Blender
# Have to make sure the selector below Vertex coordinates in Blender is set to "Global" not "Local" to copy the right coordinates.
# +x in Blender is +x in Python, lateral from right to left
# +y in Blender is -z in Python, longitudinal from head to tail
# +z in Blender is +y in Python, vertical from bottom to top

juvenile_chinook_right_eye = np.array([-0.11, 0.037, 2.82]) # Blender (x, y, z) = (-0.45, -2.91, 0.035)
juvenile_chinook_left_eye = np.array([-1, 1, 1]) * juvenile_chinook_right_eye # Eye positions started with blender coords, then modified to look good
juvenile_chinook_right_eyeball = np.array([-0.21, 0.032, 2.85]) # Based on manual tweaks to main eye positions in Python
juvenile_chinook_left_eyeball = np.array([-1, 1, 1]) * juvenile_chinook_right_eyeball
juvenile_chinook = {
    'file_path'            : '/Users/Jason/Dropbox/Drift Model Project/Calculations/3D Fish Models/JuvenileChinookSalmon.obj',\
    'model_head'           : np.array([0.0, 0.02064, 3.53901]), # Blender (x,y,z) = (0.0000, -3.53901, 0.02064)\
    'model_tail'           : np.array([0.0, -0.02884, -5.5485]), # Blender (x,y,z) = (0.0000, 5.54850, -0.02884)\
    'model_vertical'       : np.array([0, 1, 0]),\
    'eyewhite_radius'      : 0.33,\
    'eyeball_radius'       : 0.245,\
    'left_eye_center'      : juvenile_chinook_left_eye,\
    'left_eyeball_center'  : juvenile_chinook_left_eyeball,\
    'right_eye_center'     : juvenile_chinook_right_eye,\
    'right_eyeball_center' : juvenile_chinook_right_eyeball,\
}

dolly_varden_right_eye = np.array([-0.21, 0.045, 4.38]) # Blender (x, y, z) = (-0.39, -4.4, 0.045)
dolly_varden_left_eye = np.array([-1, 1, 1]) * dolly_varden_right_eye # Eye positions started with blender coords, then modified to look good
dolly_varden_right_eyeball = np.array([-0.27, 0.05, 4.39]) # Based on manual tweaks to main eye positions in Python
dolly_varden_left_eyeball = np.array([-1, 1, 1]) * dolly_varden_right_eyeball
dolly_varden = {
    'file_path'            : '/Users/Jason/Dropbox/Drift Model Project/Calculations/3D Fish Models/DollyVarden.obj',\
    'model_head'           : np.array([0.0, 0.03018, 4.94143]), # Blender (x,y,z) = (0.0000, -4.94143, 0.03018)\
    'model_tail'           : np.array([0.0, 0.00647, -4.47854]), #Blender (x,y,z) = (0.0000, 4.47854, 0.00647)\
    'model_vertical'       : np.array([0, 1, 0]),\
    'eyewhite_radius'      : 0.18,\
    'eyeball_radius'       : 0.13,\
    'left_eye_center'      : dolly_varden_left_eye,\
    'left_eyeball_center'  : dolly_varden_left_eyeball,\
    'right_eye_center'     : dolly_varden_right_eye,\
    'right_eyeball_center' : dolly_varden_right_eyeball,\
}

grayling_right_eye = np.array([-0.20, 0.04145, 4.04]) # Blender (x, y, z) = (-0.34536, -4.07590, 0.04145)
grayling_left_eye = np.array([-1, 1, 1]) * grayling_right_eye # Eye positions started with blender coords, then modified to look good
grayling_right_eyeball = np.array([-0.25, 0.05, 4.06]) # Based on manual tweaks to main eye positions in Python
grayling_left_eyeball = np.array([-1, 1, 1]) * grayling_right_eyeball
grayling = {
    'file_path'            : '/Users/Jason/Dropbox/Drift Model Project/Calculations/3D Fish Models/Grayling.obj',\
    'model_head'           : np.array([0.0, 0.00370, 4.55879]), # Blender (x,y,z) = (0.0000, -4.55879, 0.00370)\
    'model_tail'           : np.array([0.0, -0.00389, -3.77168]), # Blender (x,y,z) = (0.0000, 3.77168, -0.00389)\
    'model_vertical'       : np.array([0, 1, 0]),\
    'eyewhite_radius'      : 0.15,\
    'eyeball_radius'       : 0.11,\
    'left_eye_center'      : grayling_left_eye,\
    'left_eyeball_center'  : grayling_left_eyeball,\
    'right_eye_center'     : grayling_right_eye,\
    'right_eyeball_center' : grayling_right_eyeball,\
}

# Dicationary for looking up the right model in the function
fish_models = {'Arctic Grayling' : grayling, 'Dolly Varden' : dolly_varden, 'Chinook Salmon' : juvenile_chinook}

#############################################################################################
#
#                          FUNCTION TO ACTUALLY CREATE A FISH
#
#############################################################################################

def fish3D(head, tail, species, figure, color=(0,1,1), opacity=1.0, world_vertical = np.array([0, 0, 1]), \
    eyewhite_color = (1,1,1), eyeball_color = (0.0,0.0,0.0), eye_sphere_resolution = 50):
    """Takes numpy arrays with 3 elements for head and tail coordinates, with species specified by either
       'Arctic Grayling', 'Chinook Salmon', or 'Dolly Varden', and adds a 3-D fish graphic to 'figure' 
       with the specified colors, etc."""

    try:
        fish = fish_models[species]
    except KeyError:
        print("Called fish3d() with invalid species identifier '%s'".format(species))
    
    R = matrix_to_move_fish_head_and_tail(fish['model_head'],fish['model_tail'],fish['model_vertical'],head,tail,world_vertical)

    def relocate(source):
        filtered_source = mlab.pipeline.transform_data(source,figure=figure)
        filtered_source.filter.transform.set_matrix(R)
        filtered_source.widget.set_transform(filtered_source.filter.transform)
        filtered_source.filter.update()
        filtered_source.widget.enabled = False    
        #return source # uncomment to work with models directly in their native coordinates
        return filtered_source
        
    # Fish's body
    fish_mesh_source = mayavitools.pipeline.open(fish['file_path'],figure=figure)
    transformed_fish_mesh_source = relocate(fish_mesh_source)
    #transformed_fish_mesh_source = fish_mesh_source # Uncomment to bypass filtering and view the model directly
    mlab.pipeline.surface(transformed_fish_mesh_source,figure=figure,color=color,opacity=opacity)
    # Left eye
    lefteye = mlab.pipeline.builtin_surface()
    lefteye.source = 'sphere'
    lefteye.data_source.radius = fish['eyewhite_radius']
    lefteye.data_source.center = fish['left_eye_center']
    lefteye.data_source.phi_resolution = eye_sphere_resolution
    lefteye.data_source.theta_resolution = eye_sphere_resolution
    mlab.pipeline.surface(relocate(lefteye),figure=figure,color=eyewhite_color,opacity=opacity)
    # Right eye
    righteye = mlab.pipeline.builtin_surface()
    righteye.source = 'sphere'
    righteye.data_source.radius = fish['eyewhite_radius']
    righteye.data_source.center = fish['right_eye_center']
    righteye.data_source.phi_resolution = eye_sphere_resolution
    righteye.data_source.theta_resolution = eye_sphere_resolution
    mlab.pipeline.surface(relocate(righteye),figure=figure,color=eyewhite_color,opacity=opacity)
    # Left eyeball
    lefteyeball = mlab.pipeline.builtin_surface()
    lefteyeball.source = 'sphere'
    lefteyeball.data_source.radius = fish['eyeball_radius']
    lefteyeball.data_source.center = fish['left_eyeball_center']
    lefteyeball.data_source.phi_resolution = eye_sphere_resolution
    lefteyeball.data_source.theta_resolution = eye_sphere_resolution
    mlab.pipeline.surface(relocate(lefteyeball),figure=figure,color=eyeball_color,opacity=opacity)
    # Right eyeball
    righteyeball = mlab.pipeline.builtin_surface()
    righteyeball.source = 'sphere'
    righteyeball.data_source.radius = fish['eyeball_radius']
    righteyeball.data_source.center = fish['right_eyeball_center']
    righteyeball.data_source.phi_resolution = eye_sphere_resolution
    righteyeball.data_source.theta_resolution = eye_sphere_resolution
    mlab.pipeline.surface(relocate(righteyeball),figure=figure,color=eyeball_color,opacity=opacity)

#############################################################################################
#
#                                TEST/DEMO CODE
#
#############################################################################################

#figname="3D Fish Test Figure"
#myFig = mlab.figure(figure=figname, bgcolor = (0,0,0), size=(1024,768))
#mlab.clf(myFig)
##
##new_head = np.array([0, 0, 0])
##new_tail = np.array([1, 0, 0])
##
#new_head = np.random.rand(3)
#new_tail = np.random.rand(3)
#fish3D(new_head, new_tail, 'Dolly Varden', myFig, color=tuple(abs(np.random.rand(3))))

#
##new_head = np.random.rand(3)+(1,0,0)
##new_tail = np.random.rand(3)+(1,0,0)
##fish3D(new_head, new_tail, 'Chinook Salmon', myFig, color=tuple(abs(np.random.rand(3))))
#
##new_head = np.random.rand(3)-(1,0,0)
##new_tail = np.random.rand(3)-(1,0,0)
#fish3D(new_head, new_tail, 'Arctic Grayling', myFig, color=tuple(abs(np.random.rand(3))))