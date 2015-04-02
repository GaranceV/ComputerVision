#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"


#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>
 #include <fstream>

using namespace std;
using namespace cv;

static void help()
{
    cout << "\nThis program demonstrates the cascade recognizer. Now you can use Haar or LBP features.\n"
            "This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
            "It's most known use is for faces.\n"
            "Usage:\n"
            "./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
            "see facedetect.cmd for one call:\n"
            "./facedetect --cascade=\"../../data/haarcascades/haarcascade_frontalface_alt.xml\" \n\n"
            "During execution:\n\tHit any key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

//Prédéclaration de la méthode detectAndDraw. Paramètres : la matrice de l'image générale, le classifier recherché, et l'échelle
void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    Rect roi, Rect& previousRoi );
//Prédéclaration méthode qui vérifie les limites de la ROI. 
Rect checkRoiBoundaries(Mat img, double upXnewRoi, double upYnewRoi, double newWidth, double newHeight);
//Deux entiers pour déterminer notre ROI. 
int dx, dy;

void camShiftFct(Mat& image, Rect selection, Mat& hist);
void getTheHand(Mat image, Rect myHandRect);

int main( int argc, const char** argv ) {
    CvCapture* capture = 0;
    Mat frame, frameCopy, image;
    const string cascadeOpt = "--cascade=";
    size_t cascadeOptLen = cascadeOpt.length();
    string inputName;
    string cascadeName = "../../../data/haarcascades/haarcascade_frontalface_alt.xml";

    help();

    CascadeClassifier cascade;
	//Boucle de vérification des arguments passés en paramètre à l'exécution
    for( int i = 1; i < argc; i++ ) {
        cout << "Processing " << i << " " <<  argv[i] << endl;
        if( cascadeOpt.compare( 0, cascadeOptLen, argv[i], cascadeOptLen ) == 0 ) {
            cascadeName.assign( argv[i] + cascadeOptLen );
            cout << "  from which we have cascadeName= " << cascadeName << endl;
        }
        else if( argv[i][0] == '-' )
            cerr << "WARNING: Unknown option %s" << argv[i] << endl;
        else inputName.assign( argv[i] );
    }

    if( !cascade.load( cascadeName ) ) {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        help();
        return -1;
    }

    if( inputName.empty() || (isdigit(inputName.c_str()[0]) && inputName.c_str()[1] == '\0') ) {
        capture = cvCaptureFromCAM( inputName.empty() ? 0 : inputName.c_str()[0] - '0' );
        int c = inputName.empty() ? 0 : inputName.c_str()[0] - '0' ;
        if(!capture) cout << "Capture from CAM " <<  c << " didn't work" << endl;
    }
    else cout << "Couldn't find anything to analyse !" << endl;

    cvNamedWindow( "result", 1 );

    if( capture ) {
        cout << "In capture ..." << endl;
        Rect roi, previousroi;
        Mat computedHist;
        for(;;) {
			//On a besoin que notre Roi de départ fasse la taille de la frame de captue.
			// En var globale, le dx et dy de décalage
            IplImage* iplImg = cvQueryFrame( capture );
            frame = iplImg;
            if(frame.empty()) break;
            if(iplImg->origin == IPL_ORIGIN_TL) frame.copyTo( frameCopy );
            else flip( frame, frameCopy, 0 );
//Définition des dx et dy qui serviront au décalage de la zone d'intérêt
            //Recalculés à chaque itération... Comme la frame. En fonction de la fraction, on va + ou - vite
			dx = frameCopy.rows / 16;
			dy = frameCopy.cols / 16;
			if (previousroi.area() < 1) { //ici, on vérifie si un visage a déjà été détecté. 
				roi = Rect(Point(), frameCopy.size()); //On initialise notre ROI à la taille complète de la Frame
				detectAndDraw( frameCopy, cascade, roi, previousroi );
			}
			else {
                //La première fois, besoin de calculer l'histogramme
                //Ensite, seulement calcul de la backproj & camshift, à partir de l'histogramme
                if (computedHist.cols < 1 && computedHist.rows < 1)
                    camShiftFct(frameCopy, previousroi, computedHist);
                else camShiftFct(frameCopy, previousroi, computedHist);
            }
            if(waitKey(10) > 32) goto _cleanup_; //au dessus de 32 pour ne pas parasiter l'usage de la barre d'espace
            //Force 10ms d'attente avant un cleanup, nécessaire pour bon fonctionnement de l'affichage avec la webcam. 
        }
        waitKey(0);

_cleanup_:
        cvReleaseCapture( &capture );
    }

    cvDestroyWindow("result");

    return 0;
}

/**
*Méthode qui décale la sélection où est le visage vers le côté où sera la main. 
Prévu pour utiliser la main droite. 
*/
Rect decalageSelection(Rect selection, double largeurIm, double hauteurIm) {
Rect nvlleSelec = Rect(0, 0, selection.x, hauteurIm);
    return nvlleSelec;
}

void camShiftFct(Mat& image, Rect selection, Mat& hist) {
    Rect trackWindow;
    //déclaration matrice où on va ranger l'histogramme
    int hsize = 16;

    float hranges[] = {0,180};
    const float* phranges = hranges; //Intervalle de couleurs dans lequel on veut travailler ?

    Mat mask, hue, hsv, histimg = Mat::zeros(200, 320, CV_8UC3);

    //Transformation de l'image en nuances HSV
    cvtColor(image, hsv, COLOR_BGR2HSV);
    int _vmin = 10, _vmax = 256, smin = 30;
    //Méthode par rapport à l'intervalle de couleurs en HSV
    inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),
                        Scalar(180, 256, MAX(_vmin, _vmax)), mask);

    int ch[] = {0, 0}; //Canaux sur lesquels on va se situer
    hue.create(hsv.size(), hsv.depth()); //Matrice hue en fonction de l'image en HSV. 
    mixChannels(&hsv, 1, &hue, 1, ch, 1); // séparation de la sauturation et de hue. 

    //On déclare la matrice roi, qui est le hue de notre sélection
    //On remplit aussi la matrice maskRoi, à partir de la sélection. 
    if (hist.cols <= 1 && hist.rows <=1 ) {
        Mat roi(hue, selection), maskroi(mask, selection);
                        
        //on calcule l'histogramme & on le normalise pour minimiser les erreurs dues à rien du tout
        calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
        normalize(hist, hist, 0, 255, CV_MINMAX);

        //Attention, bloc pour vérifier que pas négatif
        histimg = Scalar::all(0);
        int binW = histimg.cols / hsize;
        Mat buf(1, hsize, CV_8UC3);
        for( int i = 0; i < hsize; i++ )
            buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
        cvtColor(buf, buf, CV_HSV2BGR);
        for( int i = 0; i < hsize; i++ ) {
            int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
            rectangle( histimg, Point(i*binW,histimg.rows),
                     Point((i+1)*binW,histimg.rows - val),
                     Scalar(buf.at<Vec3b>(i)), -1, 8 );
        }
    }

    //On choisit la trackWindow à partir de la sélection : décalage sur la droite / au visage
    trackWindow = decalageSelection(selection, image.cols, image.rows);

    Mat backproj;
    calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
    // on additionne à la matrice backproj le mask, bit à bit. 
    backproj &= mask;
    //On effectue le suivi à partir du masque & de la trackWindow
    //Dans trackbox, on met le résultat de l'application du calcul / comparaison d'histogramme sur notre zone à analyser 
    RotatedRect trackBox = CamShift(backproj, trackWindow,
                        TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));

    //ici, on récupère un rectangle tout droit à partir du rectangle de travers. 
    Rect myHandRect = trackBox.boundingRect();
    
    getTheHand(image, myHandRect);

    ellipse( image, trackBox, Scalar(0,0,255), 3, CV_AA );
    rectangle(image, selection, CV_RGB(0,255,255)); //Dessin de la région du visage

    rectangle( image, myHandRect, CV_RGB(255, 0,0)); //dessin de la région de la main
    cv::imshow( "result", image ); 

}

void getTheHand(Mat image, Rect myHandRect) {
    //Se déclenche quand on appuie sur la touche espace
if(waitKey(10) == 32){    
//Transformation Rectangle de la main en carré 
    Rect myHandSquare = checkRoiBoundaries(image, myHandRect.x, myHandRect.y, myHandRect.width, myHandRect.width);

    // on crée une matrice carrée qui contient le morceau de l'image où il devrait y avoir la main. 
    Mat emptyImg(image,myHandSquare);
    //On vide le côté droit de la matrice, où il risque le plus d'y avoir des parasites. 
        for (int j = (emptyImg.cols)*2/3 ; j < emptyImg.cols;j++) {
            for (int i = 0; i<emptyImg.rows ; i++) {
                emptyImg.at<int>(i,j) = 0;
            }
        }
        
    cv::imshow("test 1 2 1 2", emptyImg);
    printf("ecriture fichier\n");
    Mat hand2 = emptyImg;
    resize(hand2, hand2, Size(16,16), 0, 0, INTER_LINEAR);
    imshow("new hand", hand2);

    Mat img = hand2.reshape(0,1);
    ofstream os("letter.txt", ios::out | ios::app);
    os << "C,";
    for (int k = 0 ; k < img.cols - 1 ; k++) {
    os << (int)img.at<uchar>(k)<<", " ;
    }
    os << (int)img.at<uchar>(k)<<std::endl;
    os.close();
    }
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
                    Rect roi, Rect& previousRoi) {
	Mat zoneRecherche = img(roi); //Conversion du RectangleOfInterest => une matrice appelée zoneRecherche
    double t = 0;
    vector<Rect> faces; //Vecteurs de rectangles contenant des visages
    //Faces est le vecteur final qui contient les visages.
  	Mat gray, smallImg( cvRound (zoneRecherche.rows), cvRound(zoneRecherche.cols), CV_8UC1 );

/* zoneRecherche : input image
 * gray : output img
 * cv_bgr2GRAY : l'espace de couleur vers lequel on veut convertir
 * => on met l'image en niveau de gris !
 * */
    cvtColor( zoneRecherche, gray, CV_BGR2GRAY ); 
    //On resize cette nouvelle image qu'on range dans smallImg
    //scale factor horizontal axis = 0 donc =  (double)dsize.width/src.cols
    //scale factor vertical axis = 0 donc (double)dsize.height/src.rows
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    t = (double)cvGetTickCount();//pour mesurer le temps de détection
    //detectMultiScale : la méthode qui fait le facedetect
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
      //|CV_HAAR_SCALE_IMAGE,
        |CV_HAAR_FIND_BIGGEST_OBJECT, //permet d'aller + vite, & de se concentrer sur les visages + proches plutôt qu'en arrière plan. 
        Size(30, 30) );

    //Affichage de la vitesse de détection
    t = (double)cvGetTickCount() - t;
    printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );

    if (faces.size() == 0) //Dans le cas où on n'a rien trouvé, on définit une nouvelle zone d'intérêt 
          previousRoi = Rect();
	else 
    {
        Scalar faceColor = CV_RGB(0,0,255);
		Rect* r = &faces[0];
        Point center;
        int radius;

        double aspect_ratio = (double)r->width/r->height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
			//Ici, aussi bien pour le cercle que pour notre previousROI, besoin de se décaler :
			//Attention aux coords qui sont parasitées par le fait que la roi de base, est à priori au milieu de l'image, & pas en haut à gauche         center.x = cvRound((r->x + r->width*0.5) + roi.x);
            center.x = cvRound((r->x + r->width*0.5) + roi.x);
            center.y = cvRound((r->y + r->height*0.5) + roi.y);
            radius = cvRound((r->width + r->height)*0.25);
            circle( img, center, radius, faceColor, 3, 8, 0 );
        }
        else
            rectangle( img, cvPoint(cvRound(r->x), cvRound(r->y)),
                       cvPoint(cvRound((r->x + r->width-1)), cvRound((r->y + r->height-1))),
                       faceColor, 3, 8, 0); //ici, dessin cercle ou rectangle dans lequel on a trouvé un visage
 
     //On utilise le rect où le visage a été trouvé, * l'échelle, + le décalage dû au rect sur lequel on travaille. 
     /*Besoin de vérifier que la previousRoi dont on va se servir au rappel de la fonction
     * n'est pas en dehors de l'image. Si c'est le cas, je bloque le morceau qui dépasse aux limites de l'image
     * */
        double upXnewRoi = (r->x)+ roi.x - dx;
        double upYnewRoi = (r->y)+roi.y - dy;
        double newWidth = r->width + 2*dx;
        double newHeight = r->height + 2*dy;

        previousRoi = checkRoiBoundaries(img, upXnewRoi, upYnewRoi, newWidth, newHeight);
    }
    rectangle(img, previousRoi, CV_RGB(0,255,255)); //Dessin de la région d'intérêt
   		
		//fonction pour afficher l'image avec nos dessins en + sur la webcam
    cv::imshow( "result", img );
}

Rect checkRoiBoundaries(Mat img, double upXnewRoi, double upYnewRoi, double newWidth, double newHeight) {
      
     //Vérif dépasse à gauche
    if (upXnewRoi < 0)
        upXnewRoi = 0;

    //vérif dépasse en haut
    if (upYnewRoi < 0)
        upYnewRoi = 0;

    //vérif à droite
    if (upXnewRoi + newWidth > img.cols - 1)
        newWidth = img.cols - 1 - upXnewRoi;

    //vérif en bas
    if (upYnewRoi + newHeight > img.rows - 1)
        newHeight = img.rows - 1 - upYnewRoi;

    return Rect(upXnewRoi, upYnewRoi, newWidth, newHeight);
}
