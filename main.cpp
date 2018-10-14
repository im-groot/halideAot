#include <iostream>
#include "Halide.h"
using namespace Halide;

Func box_filter(ImageParam& in, Param<int >& rad)
{
    Var x("x"), y("y");

    Func clamped = BoundaryConditions::repeat_edge(in);
    Func input("input");
    input(x,y) = cast<uint32_t >(clamped(x,y));

    RDom r(-rad, 2 * rad + 1, -rad, 2 * rad + 1);
    Expr total = cast<uint32_t>((2 * rad + 1) * (2 * rad + 1));
//    total = print(total, "rad:", rad);

# if 1
    Func blur("blur");
    blur(x,y) = sum(input(x + r.x, y + r.y));

    Func output("output");
    //output(x,y) = cast<uint8_t >(print(blur(x,y), "x:", x, "y:", y, "total:", total) / total);
    output(x,y) = cast<uint8_t >(blur(x,y) / total);
//    output(x,y) = cast<uint8_t>(sum(print((input(x + r.x, y + r.y)),"x:", x + r.x, "y:", y + r.y) )
//            / ((2 * rad + 1)*(2 * rad + 1)));

//    output.trace_stores();
#else
    Func blurx("blurx");
    blurx(x,y) = sum(input(x + r.x, y));
    Func blury("blury");
    blury(x,y) = sum(blurx(x, y + r.y));

    Func output("output");
    output(x,y) = cast<uint8_t >(blury(x,y) / total);
    //output.trace_stores();
#endif

    Var xi, xo, yi, yo, tile_index;
    //output.split(y, yo, yi, 64).vectorize(yi).parallel(yo);
    output.tile(x, y, xo, yo, xi, yi, 32, 32).fuse(xo, yo, tile_index).parallel(tile_index);

    Target target = get_host_target();
    target.set_feature(Target::CUDA);

    return output;
}

Func box_filter_integral(ImageParam& in, Param<int >& rad) {
    Var x("x"), y("y");
    Func clamped = BoundaryConditions::repeat_edge(in);

    Func integral_32("integral_32");
    integral_32(x, y) = cast<int32_t>(clamped(x, y));

    RDom rx(1, in.width());
    integral_32(rx, y) += integral_32(rx - 1, y);
    //integral_32(rx.x, rx.y) += print(integral_32(rx.x - 1,rx.y), " <- b has this value when x,y is ", rx.x, ",", rx.y);

    RDom ry(1, in.height());
    integral_32(x, ry) += integral_32(x, ry - 1);
    //integral_32(ry.x, ry.y) += print(integral_32(ry.x,ry.y - 1), " <- b2 has this value when x,y is ", ry.x, ",", ry.y);

    integral_32.compute_root();
    Var xi, xo, yi, yo;
    integral_32.update(0).split(y, yo, yi, 64).vectorize(yi).parallel(yo);
    integral_32.update(1).split(x, xo, xi, 64).vectorize(xi).parallel(xo);

    Expr total = (rad * 2 + 1) * (rad * 2 + 1);
    Func blur("blur");
    blur(x, y) = (integral_32(x + rad, y + rad) + integral_32(x - rad - 1, y - rad - 1)
                  - integral_32(x - rad - 1, y + rad) - integral_32(x + rad, y - rad - 1));

    Func box_filter_integral("box_filter_integral");
    box_filter_integral(x, y) = cast<uint8_t>(blur(x, y) / total);

    Var xi2, xo2, yi2, yo2, tile_index;
    box_filter_integral.tile(x, y, xo2, yo2, xi2, yi2, 32, 32).fuse(xo2, yo2, tile_index).parallel(tile_index);

    return box_filter_integral;
}

int main() {

    Param<int> rad;
    ImageParam input(type_of<uint8_t>(), 2);

#if 0
    Func boxFilterIntegral = box_filter_integral(input, rad);
    boxFilterIntegral.compile_to_static_library("box_filter_integral", {input, rad}, "box_filter_integral");
#else
    Func boxFilter = box_filter(input, rad);
    boxFilter.compile_to_static_library("box_filter", {input, rad}, "box_filter");
#endif

    printf("Halide pipeline compiled, but not yet run¥n");

    return 0;
}