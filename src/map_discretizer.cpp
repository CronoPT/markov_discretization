
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <eigen3/Eigen/Dense>


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

#define SUCC 0.8
#define FAIL 0.2/3

#define RED   0
#define GREEN 1
#define ORANG 2
#define BLACK 3
#define OCCUP 4
#define FREE  5

#define UP    "UP"
#define DOWN  "DOWN"
#define RIGHT "RIGHT"
#define LEFT  "LEFT"

class mdp {
	std::vector<std::pair<float, float>> _states;
	std::vector<std::string> _actions = {UP, DOWN, RIGHT, LEFT};
	std::vector<Eigen::MatrixXd> _transitions;
	Eigen::MatrixXd _rewards;
	float _gamma;

	public:
	mdp(std::vector<std::pair<float, float>> states, std::vector<std::string> actions, 
	    std::vector<Eigen::MatrixXd> transitions, Eigen::MatrixXd rewards, float gamma):
		_states(states), _actions(actions), _transitions(transitions), 
		_rewards(rewards), _gamma(gamma) { /*Do Nothing*/ }

};

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
		_square_colors = new char*[discretized_grid_size_x()];
		for(int i=0; i<discretized_grid_size_x(); i++) { _square_colors[i] = new char[discretized_grid_size_y()]; }

		_pixel_colors = new char*[_map.info.width];
		for(int i=0; i<_map.info.width; i++) { _pixel_colors[i] = new char[_map.info.height]; }
	}

	~map_discretizer() {
		for(int i=0; i<discretized_grid_size_x(); i++) { delete _square_colors[i]; }
		delete _square_colors;

		for(int i=0; i<_map.info.width; i++) { delete _pixel_colors[i]; }
		delete _pixel_colors;
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

	int num_free_squares() {
		int res = 0;
		determine_square_colors();
		for(int y=discretized_grid_size_y()-1; y>=0; y--)
			for(int x=0; x<discretized_grid_size_x(); x++)
				if(_square_colors[x][y] != BLACK)
					res++;
		return res;
	}

	void compute_network() {
		determine_square_colors();

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
				if(_square_colors[x][y] != BLACK ) {
					std::cout << "| " << std::setprecision(2) << std::fixed;
					std::cout << coord_x << " " << coord_y; 
				}
				else {
					std::cout << "| OCCUPIED ";
				}
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
	}

	std::vector<std::pair<float, float>> build_states_for_mdp() {
		std::vector<std::pair<float, float>> states;
		determine_square_colors();

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
				if(_square_colors[x][y] != BLACK )
					states.push_back(std::pair<float, float>(coord_x, coord_y));
			}
		}
		return states;
	}

	std::vector<Eigen::MatrixXd> build_transitions_for_mdp() {
		std::vector<Eigen::MatrixXd> transitions;
		transitions.push_back(build_up_transition());
		transitions.push_back(build_down_transition());
		transitions.push_back(build_right_transition());
		transitions.push_back(build_left_transition());
		return transitions;	
	}

	int get_index_of_state_square(int x, int y) {
		int res = 0;
		for(int yi=discretized_grid_size_y()-1; yi>=0; yi--) {
			for(int xi=0; xi<discretized_grid_size_x(); xi++) { 
				if(xi==x && yi==y) { return res; }
				if(_square_colors[xi][yi] != BLACK) { res++; } 
			}
		}
	}

	int get_index_of_state_by_coord(float xi, float yi) {
		determine_square_colors();

		double init_square_x = 0;
		double init_square_y = 0;
		double final_square_x = 0;
		double final_square_y = 0;
		double coord_x = 0;
		double coord_y = 0;
		int pixels_per_square = _cell_size/_map.info.resolution + 1;
		int index = 0;

		for(int y=discretized_grid_size_y()-1; y>=0; y--) {
			for(int x=0; x<discretized_grid_size_x(); x++) {
				init_square_x = pixel_x_to_coordinate( _init_x + (x*pixels_per_square));
				init_square_y = pixel_y_to_coordinate( _init_y + (y*pixels_per_square));
				final_square_x = pixel_x_to_coordinate( _init_x + (x+1)*pixels_per_square);
				final_square_y = pixel_y_to_coordinate( _init_y + (y+1)*pixels_per_square);
				coord_x = (init_square_x+final_square_x)/2;
				coord_y = (init_square_y+final_square_y)/2;

				//exact comparison won't work
				if(std::fabs(coord_x-xi)<0.001 && std::fabs(coord_y-yi)<0.001){
					return index;
				}

				if(_square_colors[x][y] != BLACK) { index++; }
			}
		}

		return index;
	}

	Eigen::MatrixXd build_up_transition() {
		int squares = num_free_squares();
		Eigen::MatrixXd up(squares, squares);
		std::cout << squares << std::endl;
		float sum = 0;
		for(int y=discretized_grid_size_y()-1; y>=0; y--) {
			for(int x=0; x<discretized_grid_size_x(); x++) {
				if(_square_colors[x][y] == BLACK) { continue; }
				
				//can't go up
				if(y == discretized_grid_size_y()-1 || _square_colors[x][y+1] == BLACK) {
					sum += SUCC;
				}
				else {
					up(get_index_of_state_square(x, y), 
					   get_index_of_state_square(x, y+1)) = SUCC;
				}

				//can't go down
				if(y == 0 || _square_colors[x][y-1] == BLACK) {
					sum += FAIL;
				}
				else {
					up(get_index_of_state_square(x, y), 
					   get_index_of_state_square(x, y-1)) = FAIL;
				}

				//can't go right
				if(x == discretized_grid_size_x()-1 || _square_colors[x+1][y] == BLACK) {
					sum += FAIL;
				}
				else {
					up(get_index_of_state_square(x, y), 
					   get_index_of_state_square(x+1, y)) = FAIL;
				}

				//can't go left
				if(x == 0 || _square_colors[x-1][y] == BLACK) {
					sum += FAIL;
				}
				else {
					up(get_index_of_state_square(x, y), 
					   get_index_of_state_square(x-1, y)) = FAIL;
				}
			}
			if(_square_colors[y][y] != BLACK)
				up(get_index_of_state_square(y, y), get_index_of_state_square(y, y)) = sum;
			sum = 0;
		}
		return up;
	}

	Eigen::MatrixXd build_down_transition() {
		int squares = num_free_squares();
		Eigen::MatrixXd down(squares, squares);
		float sum = 0;
		for(int y=discretized_grid_size_y()-1; y>=0; y--) {
			for(int x=0; x<discretized_grid_size_x(); x++) {
				if(_square_colors[x][y] == BLACK) { continue; }

				//can't go up
				if(y == discretized_grid_size_y()-1 || _square_colors[x][y+1] == BLACK) {
					sum += FAIL;
				}
				else {
					down(get_index_of_state_square(x, y), 
					     get_index_of_state_square(x, y+1)) = FAIL;
				}

				//can't go down
				if(y == 0 || _square_colors[x][y-1] == BLACK) {
					sum += SUCC; 
				}
				else {
					down(get_index_of_state_square(x, y), 
					     get_index_of_state_square(x, y-1)) = SUCC;
				}

				//can't go right
				if(x == discretized_grid_size_x()-1 || _square_colors[x+1][y] == BLACK) {
					sum += FAIL; 
				}
				else {
					down(get_index_of_state_square(x, y), 
					     get_index_of_state_square(x+1, y)) = FAIL;
				}

				//can't go left
				if(x == 0 || _square_colors[x-1][y] == BLACK) {
					sum += FAIL;
				}
				else {
					down(get_index_of_state_square(x, y), 
					     get_index_of_state_square(x-1, y)) = FAIL;
				}
			}
			if(_square_colors[y][y] != BLACK)
				down(get_index_of_state_square(y, y), get_index_of_state_square(y, y)) = sum;
			sum = 0;
		}
		return down;	
	}

	Eigen::MatrixXd build_right_transition() {
		int squares = num_free_squares();
		Eigen::MatrixXd right(squares, squares);
		float sum = 0;
		for(int y=discretized_grid_size_y()-1; y>=0; y--) {
			for(int x=0; x<discretized_grid_size_x(); x++) {
				if(_square_colors[x][y] == BLACK) { continue; }

				//can't go up
				if(y == discretized_grid_size_y()-1 || _square_colors[x][y+1] == BLACK) {
					sum += FAIL;
				}
				else {
					right(get_index_of_state_square(x, y), 
					      get_index_of_state_square(x, y+1)) = FAIL;
				}

				//can't go down
				if(y == 0 || _square_colors[x][y-1] == BLACK) {
					sum += FAIL; 
				}
				else {
					right(get_index_of_state_square(x, y), 
					      get_index_of_state_square(x, y-1)) = FAIL;
				}

				//can't go right
				if(x == discretized_grid_size_x()-1 || _square_colors[x+1][y] == BLACK) {
					sum += SUCC; 
				}
				else {
					right(get_index_of_state_square(x, y),
					      get_index_of_state_square(x+1, y)) = SUCC;
				}

				//can't go left
				if(x == 0 || _square_colors[x-1][y] == BLACK) {
					sum += FAIL;
				}
				else {
					right(get_index_of_state_square(x, y), 
					      get_index_of_state_square(x-1, y)) = FAIL;
				}
			}
			if(_square_colors[y][y] != BLACK)
				right(get_index_of_state_square(y, y), get_index_of_state_square(y, y)) = sum;
			sum = 0;
		}
		return right;
	}

	Eigen::MatrixXd build_left_transition() {
		int squares = num_free_squares();
		Eigen::MatrixXd left(squares, squares);
		float sum = 0;
		for(int y=discretized_grid_size_y()-1; y>=0; y--) {
			for(int x=0; x<discretized_grid_size_x(); x++) {
				if(_square_colors[x][y] == BLACK) { continue; }

				//can't go up
				if(y == discretized_grid_size_y()-1 || _square_colors[x][y+1] == BLACK) {
					sum += FAIL;
				}
				else {
					left(get_index_of_state_square(x, y), 
					     get_index_of_state_square(x, y+1)) = FAIL;
				}

				//can't go down
				if(y == 0 || _square_colors[x][y-1] == BLACK) {
					sum += FAIL; 
				}
				else {
					left(get_index_of_state_square(x, y), 
					     get_index_of_state_square(x, y-1)) = FAIL;
				}

				//can't go right
				if(x == discretized_grid_size_x()-1 || _square_colors[x+1][y] == BLACK) {
					sum += FAIL; 
				}
				else {
					left(get_index_of_state_square(x, y),
					     get_index_of_state_square(x+1, y)) = FAIL;
				}

				//can't go left
				if(x == 0 || _square_colors[x-1][y] == BLACK) {
					sum += SUCC;
				}
				else {
					left(get_index_of_state_square(x, y), 
					     get_index_of_state_square(x-1, y)) = SUCC;
				}
			}
			if(_square_colors[y][y] != BLACK)
				left(get_index_of_state_square(y, y), get_index_of_state_square(y, y)) = sum;
			sum = 0;
		}
		return left;
	}

	Eigen::MatrixXd build_rewards_for_mdp(std::pair<float, float> goal, std::vector<std::pair<float, float>> to_avoid) {
		int squares = num_free_squares();
		Eigen::MatrixXd rewards(squares, 4);
		for(int i=0; i<rewards.cols(); i++) {
			for(int j=0; j<rewards.rows(); j++) {
				rewards(j, i) = -1;
			}
		}

		for(auto point : to_avoid) {
			int index =  get_index_of_state_by_coord(point.first, point.second);
			std::cout << index << std::endl; 
			rewards(index, 0) = -100;
			rewards(index, 1) = -100;
			rewards(index, 2) = -100;
			rewards(index, 3) = -100;
		}
		
		int goal_index = get_index_of_state_by_coord(goal.first, goal.second);
		std::cout << goal_index << std::endl; 
		rewards(goal_index, 0) = 100;
		rewards(goal_index, 1) = 100;
		rewards(goal_index, 2) = 100;
		rewards(goal_index, 3) = 100;

		return rewards;
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
	//m_d.paint_chess_filtered_map("filtered_map.ppm");
	//m_d.paint_chess_map("squared_map.ppm");
	//m_d.paint_map("normal_map.ppm");
	//m_d.print_position_of_pixel(100, 100);
	m_d.compute_network();
	//ROS_INFO("Result grid size <%d x %d>", m_d.discretized_grid_size_x(), m_d.discretized_grid_size_y());
	
	std::pair<float, float> goal(3.75, 3.90);
	std::vector<std::pair<float, float>> to_avoid;
	to_avoid.push_back(std::pair<float, float>(1.65, 0.90));
	to_avoid.push_back(std::pair<float, float>(1.65, 1.20));
	to_avoid.push_back(std::pair<float, float>(1.65, 1.50));
	to_avoid.push_back(std::pair<float, float>(1.65, 1.80));
	to_avoid.push_back(std::pair<float, float>(1.65, 2.10));
	to_avoid.push_back(std::pair<float, float>(1.65, 2.40));
	to_avoid.push_back(std::pair<float, float>(1.65, 2.70));

	std::vector<std::pair<float, float>> states = m_d.build_states_for_mdp();
	std::vector<Eigen::MatrixXd> transitions = m_d.build_transitions_for_mdp();
	Eigen::MatrixXd rewards = m_d.build_rewards_for_mdp(goal, to_avoid);
	std::vector<std::string> actions = {UP, DOWN, RIGHT, LEFT};
	float gamma = 0.9;

	mdp problem(states, actions, transitions, rewards, gamma);

	std::cout << rewards << std::endl;

	for(auto m : transitions) {
		//std::cout << m << std::endl;
		std::cout << m.rows() << " " << m.cols() << std::endl;
	}
	
	std::cout << states.size() << std::endl;

	// int i = 0;
	// for(auto p : states) {
	// 	std::cout << "State -> " << i++ << " <-> " << p.first << " -- " << p.second << std::endl;
	// }

	ROS_INFO("Finished");

	return 0;
}