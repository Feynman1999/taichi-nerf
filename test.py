import taichi as ti

ti.init(arch=ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda)

@ti.kernel 
def foo():
    a=1.7
    ans = [ ]
    b=ti.cast(a,ti.i32)
    c=ti.cast(b,ti.f32)
    ans.append(b)
    print("b=")#b=1
    print("c=")#c=1.0


foo()