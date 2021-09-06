import numpy as np
import matplotlib.pyplot as plt 

radius = 0.005
# list of odd numbers that define the plate.
plate = [17, 29, 39, 43]
m_plate = 0.01
friction = 0.8

xy_coord = []

# Dense plate
"""
for i in range(len(plate)):
    width = plate[i]
    leftmost = -int(width / 2)
    rightmost = int(width / 2)

    xy_coord.append([radius * 2.0 * leftmost, radius * 2.0 * i])
    xy_coord.append([radius * 2.0 * rightmost, radius * 2.0 * i])

    for j in range(leftmost, rightmost+1,4):
        xy_coord.append([radius * 2.0 * j, radius * 2.0 * i])

xy_coord = np.array(xy_coord)
"""

# Sparse plate.
xy_coord = []
xy_coord.append([0.0, 0.0, 0.025])
xy_coord.append([-0.05, 0.0, 0.025])
xy_coord.append([0.05, 0.0, 0.025])
xy_coord.append([-0.09, 0.008, 0.02])
xy_coord.append([0.09, 0.008, 0.02])
xy_coord.append([0.125, 0.011, 0.015])
xy_coord.append([-0.125, 0.011, 0.015])
xy_coord.append([0.15, 0.017, 0.01])
xy_coord.append([-0.15, 0.017, 0.01])
xy_coord = np.array(xy_coord)


"""
plt.figure()
plt.axis('equal')
plt.scatter(xy_coord[:,0], xy_coord[:,1])
plt.show()
"""

# 1. Initialize the file.

f = open("models/plate.sdf", 'w')
f.write("# This sdf file is auto-generated.\n")
f.write("# Manual modification is not recommended.\n")
f.write("<sdf version='1.6'>\n")
f.write("  <model name='plate'>\n")
f.write("\n")

f.write("""
    <!-- Ghost body of negligible mass. -->
    <link name="ghost_body_y">
      <inertial>
        <mass>1.0e-6</mass>
        <inertia>
          <ixx>1.0e-6</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>1.0e-6</iyy>
          <iyz>0</iyz>
          <izz>1.0e-6</izz>
        </inertia>
      </inertial>
    </link>

    <link name="ghost_body_z">
      <inertial>
        <mass>1.0e-6</mass>
        <inertia>
          <ixx>1.0e-6</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>1.0e-6</iyy>
          <iyz>0</iyz>
          <izz>1.0e-6</izz>
        </inertia>
      </inertial>
    </link>

    <joint name="y_slider" type="prismatic">
      <parent>world</parent>
      <child>ghost_body_y</child>
      <axis>
        <xyz>0 1 0</xyz>
        <!-- Drake attaches an actuator to all joints with a non-zero effort
             limit. We do not want an actuator for this joint. -->
        <limit>
          <effort>0</effort>
        </limit>
      </axis>
    </joint>

    <joint name="z_slider" type="prismatic">
      <parent>ghost_body_y</parent>
      <child>ghost_body_z</child>
      <axis>
        <xyz>0 0 1</xyz>
        <!-- Drake attaches an actuator to all joints with a non-zero effort
             limit. We do not want an actuator for this joint and therefore we
             set a zero effort limit. -->
        <limit>
          <effort>0</effort>
        </limit>
      </axis>
    </joint>

    <joint name="rotation" type="revolute">
      <parent>ghost_body_z</parent>
      <child>plate_body</child>
      <axis>
        <xyz>1 0 0</xyz>
        <!-- Drake attaches an actuator to all joints with a non-zero effort
             limit. We do not want an actuator for this joint and therefore we
             set a zero effort limit. -->
        <limit>
          <effort>0</effort>
        </limit>
      </axis>
    </joint>
""")

f.write("  <link name='plate_body'>\n")

f.write("  <pose> 0 0 0 0 0 0 </pose>\n")
f.write("  <inertial>\n")

f.write("     <mass> {:03f} </mass>\n".format(m_plate))

f.write("     <inertia>\n")
f.write("       <ixx>{:05f}</ixx>\n".format(1e-4))
f.write("       <iyy>{:05f}</iyy>\n".format(1e-4))
f.write("       <izz>{:05f}</izz>\n".format(1e-4))
f.write("       <ixy>{:05f}</ixy>\n".format(0.0))
f.write("       <iyz>{:05f}</iyz>\n".format(0.0))
f.write("       <ixz>{:05f}</ixz>\n".format(0.0))
f.write("     </inertia>\n")

f.write("  </inertial>\n")
f.write("\n")


# Add all the balls. 
for i in range(0, xy_coord.shape[0]):
    f.write("  <visual name='visual_{:03d}'>\n".format(i))
    f.write("  <pose> 0 {:03f} {:03f} 0 0 0 </pose>\n".format(
        xy_coord[i,0], xy_coord[i,1]))

    radius = xy_coord[i,2]

    f.write("  <geometry>\n")
    f.write("    <sphere>\n")
    f.write("       <radius>{:05f}</radius>\n".format(radius))
    f.write("    </sphere>\n")
    f.write("  </geometry>\n")
    f.write("  <material>\n")
    f.write("    <diffuse>1.0 1.0 1.0 1.0</diffuse>\n")
    f.write("  </material>\n")
    f.write("  </visual>\n")
    f.write("\n")

    f.write("  <collision name='collision_{:03d}'>\n".format(i))
    f.write("  <pose> 0 {:03f} {:03f} 0 0 0 </pose>\n".format(
        xy_coord[i,0], xy_coord[i,1]))
    f.write("  <geometry>\n")
    f.write("    <sphere>\n")
    f.write("       <radius>{:05f}</radius>\n".format(radius))
    f.write("    </sphere>\n")
    f.write("  </geometry>\n")
    f.write("  <surface>\n")
    f.write("    <friction>\n")
    f.write("      <ode>\n")
    f.write("        <mu>{:4f}</mu>\n".format(friction))
    f.write("        <mu2>{:4f}</mu2>\n".format(friction))
    f.write("      </ode>\n")
    f.write("    </friction>\n")
    f.write("  </surface>\n")
    f.write("  </collision>\n")

# End file.
f.write("  </link>\n\n")    
f.write("  </model>\n")
f.write("</sdf>\n")
