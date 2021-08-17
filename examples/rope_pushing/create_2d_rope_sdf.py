import numpy as np

radius = 0.01
num_balls = 30

m_ball = 0.001 
# 2/5 for solid, 2/3 for thin shell.
inertia_ball = 2/5 * m_ball * (radius ** 2.0)
friction = 0.8

# 1. Initialize the file.

f = open("models/rope.sdf", 'w')
f.write("# This sdf file is auto-generated.\n")
f.write("# Manual modification is not recommended.\n")
f.write("<sdf version='1.6'>\n")
f.write("  <model name='rope'>\n")
f.write("\n")

# Add all the balls. 
for i in range(0, num_balls):
    f.write("  <link name='link_{:03d}'>\n".format(i))

    f.write("  <pose> 0 0 {:03f} 0 0 0 </pose>\n".format(-i * 2.0 * radius))
    f.write("  <inertial>\n")

    f.write("     <mass> {:03f} </mass>\n".format(m_ball))

    f.write("     <inertia>\n")
    f.write("       <ixx>{:05f}</ixx>\n".format(inertia_ball))
    f.write("       <iyy>{:05f}</iyy>\n".format(inertia_ball))
    f.write("       <izz>{:05f}</izz>\n".format(inertia_ball))
    f.write("       <ixy>{:05f}</ixy>\n".format(0.0))
    f.write("       <iyz>{:05f}</iyz>\n".format(0.0))
    f.write("       <ixz>{:05f}</ixz>\n".format(0.0))
    f.write("     </inertia>\n")
    
    f.write("  </inertial>\n")
    f.write("\n")

    f.write("  <visual name='link_{:03d}_1'>\n".format(i))
    f.write("  <pose> 0 0 0 0 0 0 </pose>\n".format(-i * 2.0 * radius))

    f.write("  <geometry>\n")
    f.write("    <sphere>\n")
    f.write("       <radius>{:05f}</radius>\n".format(radius))
    f.write("    </sphere>\n")
    f.write("  </geometry>\n")
    f.write("  <material>\n")
    f.write("    <diffuse>{:05f} 0.5 1.0 0.7</diffuse>\n".format(i / num_balls))
    f.write("  </material>\n")
    f.write("  </visual>\n")
    f.write("\n")


    f.write("  <collision name='collision_{:03d}'>\n".format(i))
    f.write("  <pose> 0 0 0 0 0 0 </pose>\n".format(-i * 2.0 * radius))    
    f.write("  <geometry>\n")
    f.write("    <sphere>\n")
    f.write("       <radius>{:05f}</radius>\n".format(radius - 0.001))
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
    f.write("  </link>\n\n")    

f.write("""
  <joint name='joint_weld' type='fixed'>
      <parent>world</parent>
      <child>link_000</child>
  </joint>

""")

# Add all the joints.
for i in range(num_balls-1):
    f.write("  <joint name='joint_{:03d}' type='revolute'>\n".format(i))
    f.write("    <parent>link_{:03d}</parent>\n".format(i))
    f.write("    <child>link_{:03d}</child>\n".format(i+1))
    f.write("    <pose> 0 0 {:03f} 0 0 0 </pose>".format(radius * 2.0))
    f.write("    <axis>\n")
    f.write("      <xyz expressed_in='__model__'> 1 0 0 </xyz>\n")
    f.write("    </axis>\n")
    f.write("  </joint>\n\n")

# End file.

f.write("  </model>\n")
f.write("</sdf>\n")
