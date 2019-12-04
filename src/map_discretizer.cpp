
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <utility>

#include "ros/ros.h"
#include "ros/console.h"

#include "nav_msgs/GetMap.h"
#include "nav_msgs/OccupancyGrid.h"
#include "nav_msgs/MapMetaData.h"
#include "std_msgs/Int8.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Quaternion.h"

#include "nav_msgs/GetMap.h"

#define RED   0
#define GREEN 1
#define ORANG 2
#define BLACK 3
#define OCCUP 4
#define FREE  5

class map_discretizer {
	nav_msgs::OccupancyGrid _map;
	int _init_x;
	int _init_y;
	int _final_x;
	int _final_y;
	double _cell_size;
	int _scale;
	char** _square_colors;
	char** _pixel_colors;

	void compute_init_x() {
		for(int x=0; x<_map.info.width; x++)
			for(int y=0; y<_map.info.height; y++)
				if (_map.data[y*_map.info.height+x] == 0) {
					_init_x  = x;
					return;
				}
	}

	void compute_init_y() {
		for(int y=0; y<_map.info.height; y++) 
			for(int x=0; x<_map.info.width; x++) 
				if (_map.data[y*_map.info.height+x] == 0) {
					_init_y = y;
					return;
				}
	}

	void compute_final_x() {
		for(int x=_map.info.width-1; x>=0; x--)
			for(int y=0; y<_map.info.height; y++)
				if (_map.data[y*_map.info.height+x] == 0) {
					_final_x  = x;
					return;
				}
	}

	void compute_final_y() {
		for(int y=_map.info.height-1; y>=0; y--)
			for(int x=0; x<_map.info.width; x++)
				if (_map.data[y*_map.info.height+x] == 0) {
					_final_y = y;
					return;
				}
	}

	bool point_within_bounds(int x, int y) {
		return (x>=_init_x && x<=_final_x && y>=_init_y && y<=_final_y);
	}

	public:
	map_discretizer(nav_msgs::OccupancyGrid map, double cell_size, int scale):
		_map(map), _init_x(-1), _init_y(-1),
		_final_x(-1), _final_y(-1), _cell_size(cell_size), _scale(scale),
		_square_colors(nullptr), _pixel_colors(nullptr) { 
		compute_init_x();
		compute_init_y();
		compute_final_x();
		compute_final_y();
	}

	void print_info() {
		ROS_INFO("Pose of origin -> (%.3f, %.3f)", _map.info.origin.position.x, _map.info.origin.position.y);
	}

	void print_position_of_pixel(int x, int y) {
		double world_x = x*_map.info.resolution - _map.info.origin.position.x;
		double world_y = y*_map.info.resolution - _map.info.origin.position.y;
		ROS_INFO("Pixel (%d, %d) corresponds to position (%.3f, %.3f) in the map", x, y, world_x, world_y);
	}

	double pixel_x_to_coordinate(int x) {
		return x*_map.info.resolution + _map.info.origin.position.x;
	}

	double pixel_y_to_coordinate(int y) {
		return y*_map.info.resolution + _map.info.origin.position.y;
	}

	int discretized_grid_size_x() {
		return (_final_x - _init_x)*_map.info.resolution/_cell_size + 1;
	}

	int discretized_grid_size_y() {
		return (_final_y - _init_y)*_map.info.resolution/_cell_size + 1;
	}

	void compute_network() {
		std::vector<std::pair<int, int>> states;

		double init_square_x = 0;
		double init_square_y = 0;
		double final_square_x = 0;
		double final_square_y = 0;
		double coord_x = 0;
		double coord_y = 0;
		int pixels_per_square = _cell_size/_map.info.resolution + 1;
		int i = 0;


		for(int y=discretized_grid_size_y()-1; y>=0; y--) {
			for(int x=0; x<discretized_grid_size_x(); x++, i++) {
				init_square_x = pixel_x_to_coordinate( _init_x + (x*pixels_per_square));
				init_square_y = pixel_y_to_coordinate( _init_y + (y*pixels_per_square));
				final_square_x = pixel_x_to_coordinate( _init_x + (x+1)*pixels_per_square);
				final_square_y = pixel_y_to_coordinate( _init_y + (y+1)*pixels_per_square);
				coord_x = (init_square_x+final_square_x)/2;
				coord_y = (init_square_y+final_square_y)/2;
				std::cout << "|" << std::setprecision(2) << std::fixed;
				std::cout << coord_x << " " << coord_y; 
			}
			std::cout << std::endl;
		}

	}

	void paint_map(const std::string& path) {
		std::ofstream img(path);
		img << "P3" << std::endl;
		img << _map.info.width*_scale << " " << _map.info.height*_scale << std::endl;
		img << "255" << std::endl; 
		for(int y=_map.info.height-1; y>=0; y--) {
			for(int sy=0; sy<_scale; sy++)
			for(int x=0; x<_map.info.width; x++)
			for(int sx=0; sx<_scale; sx++) {
				if(x==_init_x && y==_init_y) {
					img << 0 << " " << 0 << " " << 255 << std::endl;
				}
				else if(x==_final_x && y==_final_y) {
					img << 255 << " " << 0 << " " << 0 << std::endl;
				}
				else {
					if(_map.data[y*_map.info.height+x] == -1) {
						img << 235 << " " << 158 << " " << 52 << std::endl;
					}
					else if (_map.data[y*_map.info.height+x] == 0) {
						img << 255 << " " << 255 << " " << 255 << std::endl;
					}
					else if (_map.data[y*_map.info.height+x] == 100) {
						img << 0 << " " << 0 << " " << 0 << std::endl;
					}
				}
			}
		}
	}

	void paint_chess_map(const std::string& path) {
		std::ofstream img(path);
		img << "P3" << std::endl;
		img << _map.info.width*_scale << " " << _map.info.height*_scale << std::endl;
		img << "255" << std::endl; 
		ROS_INFO("Size of each cell in pixels -> %d", (int)(_cell_size/_map.info.resolution + 1));

		for(int y=_map.info.height-1; y>=0; y--) {
			for(int sy=0; sy<_scale; sy++)
			for(int x=0; x<_map.info.width; x++) 
			for(int sx=0; sx<_scale; sx++) {
				if(point_within_bounds(x, y)) {
					if(_map.data[y*_map.info.height+x] == 0) {
						if( (((int)(x-_init_x))%((int) (2*_cell_size/_map.info.resolution + 1))) > _cell_size/_map.info.resolution ) {
							if( (((int)(y-_init_y))%((int) (2*_cell_size/_map.info.resolution + 1))) > _cell_size/_map.info.resolution )
								img << 117 << " " << 255 << " " << 168 << std::endl;
							else
								img << 255 << " " << 117 << " " << 117 << std::endl;
						}
						else {
							if( (((int)(y-_init_y))%((int) (2*_cell_size/_map.info.resolution + 1))) < _cell_size/_map.info.resolution )
								img << 117 << " " << 255 << " " << 168 << std::endl;
							else
								img << 255 << " " << 117 << " " << 117 << std::endl;
						}
					}
					else if(_map.data[y*_map.info.height+x] == -1) {
						img << 235 << " " << 158 << " " << 52 << std::endl;
					}
					else if (_map.data[y*_map.info.height+x] == 100) {
						img << 0 << " " << 0 << " " << 0 << std::endl;
					}
				}
				else if(_map.data[y*_map.info.height+x] == -1) {
					img << 235 << " " << 158 << " " << 52 << std::endl;
				}
				else if (_map.data[y*_map.info.height+x] == 100) {
					img << 0 << " " << 0 << " " << 0 << std::endl;
				}
			}
		}
	}

	void determine_square_colors() {
		for(int y=discretized_grid_size_y()-1; y>=0; y--)
			for(int x=0; x<discretized_grid_size_x(); x++)
				_square_colors[x][y] = determine_color_of_square(x, y);
	}

	void determine_pixel_colors() {
		for(int x=0; x<_map.info.width; x++) {
			for(int y=0; y<_map.info.height; y++) {
				if(point_within_bounds(x, y)) {
					auto square = determine_square_of_pixel(x, y);
					int x_square = square.first;
					int y_square = square.second;
					if(_square_colors[x_square][y_square]==BLACK) {
						_pixel_colors[x][y] = BLACK;
					}
					else if(x_square%2==0 && y_square%2==0) {
						_pixel_colors[x][y] = RED;
					}
					else if(x_square%2==0 && y_square%2!=0) {
						_pixel_colors[x][y] = GREEN;
					}
					else if(x_square%2!=0 && y_square%2!=0) {
						_pixel_colors[x][y] = RED;
					}
					else if(x_square%2!=0 && y_square%2==0) {
						_pixel_colors[x][y] = GREEN;
					}
				}
				else if(_map.data[y*_map.info.height+x] == -1) {
					_pixel_colors[x][y] = ORANG;
				}
				else if(_map.data[y*_map.info.height+x] == 100) {
					_pixel_colors[x][y] = BLACK;
				}
			}
		}
	}

	std::pair<int, int> determine_square_of_pixel(int x, int y) {
		int x_delocation = x-_init_x;
		int y_delocation = y-_init_y;
		int x_square = x_delocation/(((double)_cell_size/_map.info.resolution));
		int y_square = y_delocation/(((double)_cell_size/_map.info.resolution));
		std::pair<int, int> res(x_square, y_square);
		return res;
	}

	//FIXME
	int determine_color_of_square(int x, int y) {
		int pixels_per_square = _cell_size/_map.info.resolution + 1;
		int init_square_x = _init_x + (x*pixels_per_square);
		int init_square_y = _init_y + (y*pixels_per_square);
		int final_square_x = _init_x + (x+1)*pixels_per_square;
		int final_square_y = _init_y + (y+1)*pixels_per_square;
		double coord_x = 0;
		double coord_y = 0;
		for(int i=init_square_x; i<final_square_x; i++) {
			for(int j=init_square_y; j<final_square_y; j++) {
				if(_map.data[j*_map.info.height+i] == -1 || _map.data[j*_map.info.height+i] == 100 )
					return BLACK;
			}
		}
		return FREE;
	}

	void paint_chess_filtered_map(const std::string& path) {
		_square_colors = new char*[discretized_grid_size_x()];
		for(int i=0; i<discretized_grid_size_x(); i++) { _square_colors[i] = new char[discretized_grid_size_y()]; }

		_pixel_colors = new char*[_map.info.width];
		for(int i=0; i<_map.info.width; i++) { _pixel_colors[i] = new char[_map.info.height]; }

		determine_square_colors();
		determine_pixel_colors();

		std::ofstream img(path);
		img << "P3" << std::endl;
		img << _map.info.width*_scale << " " << _map.info.height*_scale << std::endl;
		img << "255" << std::endl; 
		ROS_INFO("Size of each cell in pixels -> %d", (int)(_cell_size/_map.info.resolution + 1));	

		for(int y=_map.info.height-1; y>=0; y--) {
			for(int sy=0; sy<_scale; sy++)
			for(int x=0; x<_map.info.width; x++) 
			for(int sx=0; sx<_scale; sx++) {
				if(_pixel_colors[x][y] == BLACK) {
					img << 0 << " " << 0 << " " << 0 << std::endl;
				}
				else if(_pixel_colors[x][y] == ORANG) {
					img << 235 << " " << 158 << " " << 52 << std::endl;
				}
				else if(_pixel_colors[x][y] == RED) {
					img << 117 << " " << 255 << " " << 168 << std::endl;
				}
				else if(_pixel_colors[x][y] == GREEN) {
					img << 255 << " " << 117 << " " << 117 << std::endl;
				}
			}
		}

		for(int i=0; i<discretized_grid_size_x(); i++) { delete _square_colors[i]; }
		delete _square_colors;

		for(int i=0; i<_map.info.width; i++) { delete _pixel_colors[i]; }
		delete _pixel_colors;
	}

};

int main(int argc, char* argv[]) {
	ros::init(argc, argv, "map_discretizer");

	ros::NodeHandle n;

	nav_msgs::GetMap::Request  req;
	nav_msgs::GetMap::Response resp;
	ROS_INFO("Requesting the map...");
	while(!ros::service::call("static_map", req, resp)) {
		ROS_WARN("Request for map failed; trying again...");
		ros::Duration d(0.5);
		d.sleep();
	}

	ROS_INFO("Received");
	
	map_discretizer m_d(resp.map, 0.3, 5);
	m_d.paint_chess_filtered_map("filtered_map.ppm");
	m_d.paint_chess_map("squared_map.ppm");
	m_d.paint_map("normal_map.ppm");
	m_d.print_info();
	m_d.print_position_of_pixel(100, 100);
	ROS_INFO("Result grid size <%d x %d>", m_d.discretized_grid_size_x(), m_d.discretized_grid_size_y());


	ROS_INFO("Finished");

	return 0;
}