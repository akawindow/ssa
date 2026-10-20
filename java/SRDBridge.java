// Project-owned I/O adapter. The SRD implementation is loaded from the unmodified author JAR.
import Algorithm.SRD;
import java.io.*;
import java.nio.file.*;
import java.util.Locale;

public class SRDBridge {
    private static final int MAGIC=0x53524431;

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        if(args.length==0) throw new IllegalArgumentException("Expected repair, smoke-fixture, or real-pair");
        switch(args[0]) {
            case "repair" -> repairBatch(args);
            case "smoke-fixture" -> smokeFixture(args);
            case "real-pair" -> realPair(args);
            default -> throw new IllegalArgumentException("Unknown mode: "+args[0]);
        }
    }

    private static void repairBatch(String[] args) throws Exception {
        if(args.length!=6) throw new IllegalArgumentException("repair input output period k iterations");
        int period=Integer.parseInt(args[3]);double k=Double.parseDouble(args[4]);int iterations=Integer.parseInt(args[5]);
        if(period<2 || period%2!=0 || !(k>0) || !Double.isFinite(k) || iterations<1) throw new IllegalArgumentException("Invalid parameters");
        try(DataInputStream in=new DataInputStream(new BufferedInputStream(Files.newInputStream(Path.of(args[1]))));
            DataOutputStream out=new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(Path.of(args[2]),StandardOpenOption.CREATE_NEW)))) {
            if(in.readInt()!=MAGIC) throw new IOException("Invalid input magic");
            int rows=in.readInt(),length=in.readInt();
            if(rows<1 || length<2*period) throw new IOException("Invalid matrix dimensions");
            out.writeInt(MAGIC);out.writeInt(rows);out.writeInt(length);
            for(int row=0;row<rows;row++) {
                double[] x=new double[length];long[] time=new long[length];
                for(int i=0;i<length;i++) {x[i]=in.readDouble();time[i]=i;if(!Double.isFinite(x[i]))throw new IOException("Nonfinite input");}
                double[] repaired=new SRD(time,x,period,k,iterations).getTd_repair();
                for(double value:repaired) {if(!Double.isFinite(value))throw new IOException("Nonfinite SRD output");out.writeDouble(value);}
            }
            if(in.read()!=-1)throw new IOException("Trailing input bytes");
        }
    }

    private static void smokeFixture(String[] args) throws Exception {
        if(args.length!=3)throw new IllegalArgumentException("smoke-fixture bank_csv output_csv");
        LoadData data=new LoadData(args[1],2000);
        double[] clean=data.getTd_clean();long[] time=data.getTd_time();
        double[] dirty=new AddNoise(clean,5.0,2.0,25,666).getTd_dirty();
        double[] repaired=new SRD(time,dirty,288,6.0,5).getTd_repair();
        writeFixture(Path.of(args[2]),time,clean,dirty,repaired);
    }

    private static void realPair(String[] args) throws Exception {
        if(args.length!=7)throw new IllegalArgumentException("real-pair clean_csv dirty_csv period k iterations output_csv");
        LoadData cleanData=new LoadData(args[1],Integer.MAX_VALUE);
        LoadData dirtyData=new LoadData(args[2],Integer.MAX_VALUE);
        double[] clean=cleanData.getTd_clean(),dirty=dirtyData.getTd_clean();
        long[] time=cleanData.getTd_time(),dirtyTime=dirtyData.getTd_time();
        if(clean.length!=dirty.length)throw new IllegalArgumentException("Pair length mismatch");
        for(int i=0;i<time.length;i++)if(time[i]!=dirtyTime[i])throw new IllegalArgumentException("Pair timestamp mismatch");
        double[] repaired=new SRD(time,dirty,Integer.parseInt(args[3]),Double.parseDouble(args[4]),Integer.parseInt(args[5])).getTd_repair();
        writeFixture(Path.of(args[6]),time,clean,dirty,repaired);
    }

    private static void writeFixture(Path path,long[] time,double[] clean,double[] dirty,double[] repaired)throws Exception {
        try(BufferedWriter out=Files.newBufferedWriter(path,StandardOpenOption.CREATE_NEW)) {
            out.write("timestamp,clean,dirty,repaired\n");
            for(int i=0;i<clean.length;i++)out.write(time[i]+","+clean[i]+","+dirty[i]+","+repaired[i]+"\n");
        }
    }
}
