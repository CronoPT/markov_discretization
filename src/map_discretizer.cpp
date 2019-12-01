
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

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

class map_discretizer {
	nav_msgs::OccupancyGrid _map;
	int _init_x;
	int _init_y;
	int _final_x;
	int _final_y;
	double _cell_size;

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
	map_discretizer(nav_msgs::OccupancyGrid map, double cell_size):
		_map(map), _init_x(-1), _init_y(-1),
		_final_x(-1), _final_y(-1), _cell_size(cell_size) { 
		compute_init_x();
		compute_init_y();
		compute_final_x();
		compute_final_y();
	}

	void paint_map(const std::string& path) {
		std::ofstream img(path);
		img << "P3" << std::endl;
		img << _map.info.width << " " << _map.info.height << std::endl;
		img << "255" << std::endl; 
		for(int y=_map.info.height-1; y>=0; y--) {
			for(int x=0; x<_map.info.width; x++) {
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
		img << _map.info.width << " " << _map.info.height << std::endl;
		img << "255" << std::endl; 
		ROS_INFO("Size of each cell in pixels -> %d", (int)(_cell_size/_map.info.resolution + 1));

		for(int y=_map.info.height-1; y>=0; y--) {
			for(int x=0; x<_map.info.width; x++) {
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

};

int main(int argc, char* argv[]) {
	ros::init(argc, argv, "map_discretizer");

	ros::NodeHandle n;

	nav_msgs::GetMap::Request  req;
	nav_msgs::GetMap::Response resp;
	ROS_INFO("Requesting the map...");
	while(!ros::service::call("static_map", req, resp))
	{
		ROS_WARN("Request for map failed; trying again...");
		ros::Duration d(0.5);
		d.sleep();
	}

	ROS_INFO("Received");
	
	map_discretizer m_d(resp.map, 0.3);
	m_d.paint_chess_map("mapina.ppm");

	ROS_INFO("Finished");

	return 0;
}