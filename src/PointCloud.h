int			max_radius = 30;
int			inf_border = 160;		// Range (in pixel) from the pole to exclude from point cloud generation 
CvSize		img_size;
double		unit_h, unit_w;	//angular size of 1 pixel
float		baseline;			//distance between two cameras (input parameter). DWRC, usability, Listening: 0.264 / Studio: 0.202 / Courtyard: 0.287 / Kitchen: 0.176
float		disp_scale = 2;
float		disp_offset = -120;

unit_h = 1.0 / (img_size.height);
unit_w = 2.0 / (img_size.width);

CvPoint3D32f vertex_point(int row, int col)
{
	CvPoint3D32f point_3D = cvPoint3D32f(0, 0, 0);
	double longitude, latitude, radius, angle_disp;

	latitude = row*unit_h * CV_PI;
	longitude = col*unit_w * CV_PI;

	if (disparity.at<float>(row, col) == 0)
		return point_3D;

	angle_disp = (disparity.at<float>(row, col) / disp_scale + disp_offset) * unit_h * CV_PI;

	if (latitude + angle_disp <0)
		angle_disp = 0.01;

	if (angle_disp == 0)
	{
		radius = max_radius;
		disparity.at<float>(row, col) = 0;
	}
	else
		radius = baseline / ((sin(latitude) / tan(latitude + angle_disp)) - cos(latitude));

	if (radius > max_radius || radius < 0.0)
	{
		radius = max_radius;
		disparity.at<float>(row, col) = 0;
	}

	point_3D.x = radius*sin(latitude)*cos(CV_PI - longitude);
	point_3D.y = radius*sin(latitude)*sin(CV_PI - longitude);
	point_3D.z = radius*cos(latitude);

	return point_3D;
}

int point_cloud_generation()
{
	int total_v;
	int i, j, k;
	CvPoint3D32f point_3D;
	int num_vertex = 0;
	
	for (i = 0; i< img_size.height; i++)
		for (j = 0; j < img_size.width; j++)
		{
			if (disparity.at<float>(i, j) == 0)
				continue;
			if (i<inf_border || i> img_size.height - inf_border)
				continue;

			point_3D = vertex_point(i, j);
			if (point_3D.x == 0 && point_3D.y == 0 && point_3D.z == 0)
				continue;

			vertex[num_vertex].x = point_3D.x;
			vertex[num_vertex].y = point_3D.y;
			vertex[num_vertex].z = point_3D.z;
			num_vertex++;
		}

	return num_vertex;
}