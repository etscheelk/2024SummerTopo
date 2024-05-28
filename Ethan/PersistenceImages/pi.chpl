
use IO;
use Random;

writeln("hello world!");

const numElems : uint(32) = 100;

var arr : [0..#numElems, 1..2];

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
        
        arr[row, 2] = pers;
    }
}

reshapePers(arr);

writeln(arr);

const numPixels : uint(32) = 1024;

var img : [0..#numPixels, 0..#numPixels] real(64);

