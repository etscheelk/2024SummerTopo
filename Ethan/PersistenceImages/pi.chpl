
use IO;
use Random;
use Math;

use image;

writeln("hello world!");

config const numElems : uint(32) = 100;

var arr : [0..#numElems, 1..2] real(64);

Random.fillRandom(arr);

forall i in arr.domain.dim[0]
{
    var x = arr[i, 1];
    var y = arr[i, 2];

    y = y * (1 - x) + x;
    arr[i, 2] = y;
}

// writeln(arr);

proc reshapePers(ref arr) : void
{
    forall row in arr.domain.dim[0]
    {
        var x = arr[row,1];
        var y = arr[row,2];

        var pers = y - x;
        pers = (y-x) / (1-x);
        
        arr[row, 2] = pers;
    }
}

reshapePers(arr);

writeln(arr);

const numPixels : uint(32) = 1024;

var img : [0..#numPixels, 0..#numPixels] real(64);


proc norm(x1, y1, x2, y2) : real(64)
{
    var x3 = x2 - x1;
    var y3 = y2 - y1;

    var front = 1 / (2 * pi);
    var rest = exp(-0.5 * (x3 * x3 + y3 * y3));

    return front * rest;
}

proc pixelize(const dim : uint, const pers, rad : real = 1) : [0..#dim, 0..#dim] real
{
    var a : [0..#dim, 0..#dim] real;
    var dim_f = dim:real;

    for (i, j) in a.domain
    {
        var i_f = i / dim_f * rad;
        var j_f = j / dim_f * rad;

        var thisSpot : real = 0;

        forall row in pers.domain.dim[0]
        with (+ reduce thisSpot)
        {
            thisSpot += norm(pers[row,2] * rad, pers[row,1] * rad, j_f, i_f);
        }

        if (j == 0)
        {
            writeln("pos ", i, ": ", thisSpot);
        }

        a[i, j] = thisSpot;
    }

    return a;
}

config const rad    : real = 10;
config const mult   : real = 120;
config const dim    : uint = 128;

var pix = pixelize(dim, arr, rad) * mult;
writeln();
writeln(arr[..,1]);

var fw = IO.openWriter("./out/out.bmp", locking = false);

image.writeImageBMP(fw, pix);

fw.close();


// test
var A = [1 => "one", 10 => "ten", 3 => "three", 16 => "sixteen"];
writeln(A.type:string);

// var B = 