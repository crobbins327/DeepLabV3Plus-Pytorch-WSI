import qupath.lib.regions.*
import ij.*
import java.awt.Color
import java.awt.*
import java.awt.geom.Area
import java.awt.image.BufferedImage
import javax.imageio.ImageIO

// Read RGB image & show in ImageJ (won't work for multichannel!)
double downsample = 1
//double downsample = 0.2/0.125
def server = getCurrentImageData().getServer()
int w = (server.getWidth() / downsample) as int
int h = (server.getHeight() / downsample) as int
def img = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)


def g2d = img.createGraphics()
g2d.scale(1.0/downsample, 1.0/downsample)
g2d.setColor(new Color (0,0,0))
g2d.setBackground(new Color (0,0,0));
g2d.clearRect(0,0,(int)w,(int)h);


//classes = ['Glomerulus', 'Glomerular Tuft', 
//            'PCT', 'DCT', 
//            'PeriTubCap', 'Artery', 'Vein',
//            'Tubule_Thy', 'Tubule_Atrophy', 'Tubule_Simp',
//            'Whitespace']

//classes = ['Ignore']

//classes = ['Other', 'Lymphocyte', 'Mitosis']
def classes = [
            "Use Patch" : 0,
            "Lymphocytes" : 1,
            "Neutrophils" : 2,
            "Macrophage" : 3,
            "PCT Nuclei" : 4,
            "DCT Nuclei" : 5,
            "Endothelial" : 6,
            "Fibroblast" : 7,
            "Mesangial" : 8,
            "Parietal cells" : 9,
            "Podocytes" : 10,
            "Mitosis" : 11,
            "Tubule Nuclei" : 12,

            // "Myofibroblast" : 12,
            // "Smooth muscle" : 13,
            // "Partial Nuclei" : 14,
            // "Tubule Nuclei" : 15,
            // "Glomerular Nuclei" : 16,
            // "Interstitial Nuclei" : 17,
            // "Vascular Nuclei" : 18,
//            "Other",
//            "Ignore*",
//            "Immune cells", 
            ]
//classes = ['Use Patch']
int total_classes = classes.size()
println(total_classes)
class_keys = classes.keySet() as String[];
for (int i =0; i<classes.size(); i++){
    selectObjects { p -> p.getPathClass() == getPathClass(class_keys[i])};
    int c = i+1
    //int c = 255
    int val = Math.round(255/total_classes*c) & 0xff;
    val = val - 1
    println('Color: '+ val)
    classes[class_keys[i]] = val;
    int count = 0
    for (a in getSelectedObjects()) {
        roi = a.getROI()
        def shape = roi.getShape()
        g2d.setPaint(new Color(val, val, val))
        g2d.fill(shape)
        if (i > 0){
            g2d.setStroke(new BasicStroke(2)) // Set the stroke width as needed
            g2d.setPaint(new Color(255, 255, 255)) // White color for maximum intensity for outline
            g2d.draw(shape)
        }
        count = count + 1
    }
    println(class_keys[i] + ' count ' +count)
}
println(classes)
g2d.dispose()
new ImagePlus("Mask", img).show()
def name = getProjectEntry().getImageName()[0..-5] //+ '.tif'
//def num = getProjectEntry().getImageName()[0..-5]
//def name = 'EMT6#2_p400_s25-lvl' + num
def path = buildFilePath(PROJECT_BASE_DIR, 'cell-masks')
mkdirs(path)
println(name+' results exporting...')
def fileoutput = new File( path, name+ '-cell-mask-'+downsample.toString()+'ds.png')
ImageIO.write(img, 'png', fileoutput)
//Save classes file as json
def gson = GsonTools.getInstance(true)
File jsonFile = new File(path, 'classes.json')
jsonFile.withWriter('UTF-8') {
     gson.toJson(classes,it)
 }