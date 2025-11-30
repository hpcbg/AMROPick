module generate(name, scales=[1,1,1], offsets=[0,0,0], roll=0, pitch=0, z=0, legs=[[15,10], [75,20], [130,10]]) {
    difference() {
        union() {
            translate(offsets) scale(scales) import(name);
            for (leg = legs) {
                translate([leg[0],leg[1],-50]) cube([7,7,50]);
            }
        }
        translate([0,0,-z]) rotate([0,pitch,0]) rotate([-roll,0,0]) rotate([0,90,0]) cube([200,200,200]);
    }
}


//generate("Plate3_original.stl",[0.5,0.5,1],[75,0,0],roll=0,pitch=0,z=0);
//generate("Plate3_original.stl",[0.5,0.5,1],[75,0,0],roll=0,pitch=0,z=5);
//generate("Plate3_original.stl",[0.5,0.5,1],[75,0,0],roll=0,pitch=0,z=10);
//generate("Plate3_original.stl",[0.5,0.5,1],[75,0,0],roll=15,pitch=0,z=0);
//generate("Plate3_original.stl",[0.5,0.5,1],[75,0,0],roll=30,pitch=0,z=0);
//generate("Plate3_original.stl",[0.5,0.5,1],[75,0,0],roll=0,pitch=15,z=0);
generate("Plate3_original.stl",[0.5,0.5,1],[75,0,0],roll=30,pitch=15,z=0);