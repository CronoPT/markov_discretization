
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <eigen3/Eigen/Dense>
#include <math.h>


#include "ros/ros.h"
#include "ros/console.h"

#include "move_base_msgs/MoveBaseAction.h"
#include "actionlib/client/simple_action_client.h"

#include "nav_msgs/GetMap.h"
#include "nav_msgs/OccupancyGrid.h"
#include "nav_msgs/MapMetaData.h"
#include "std_msgs/Int8.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Quaternion.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"

#include "nav_msgs/GetMap.h"

#define SUCC 0.79
#define FAIL 0.07

#define RED1  0
#define RED2  11
#define BLUE1 12
#define BLUE2 13
#define GREEN 1
#define ORANG 2
#define BLACK 3
#define OCCUP 4
#define FREE  5

#define UP    "UP"
#define DOWN  "DOWN"
#define RIGHT "RIGHT"
#define LEFT  "LEFT"

#define DEADLY -100
#define NORMAL -1
#define GOAL   100

double __x = 0;
double __y = 0;

void chatterCallback(const geometry_msgs::PoseWithCovarianceStamped& msg) {
	__x = msg.pose.pose.position.x;
	__y = msg.pose.pose.position.y;
}


/*===================================================================
| mdp -> a class to represent mdps
===================================================================*/
class mdp {
	std::vector<std::pair<float, float>> _states;
	std::vector<std::string> _actions;
	std::vector<Eigen::MatrixXd> _transitions;
	Eigen::MatrixXd _rewards;
	double _gamma;

	public:
	mdp(std::vector<std::pair<float, float>> states, std::vector<std::string> actions, 
	    std::vector<Eigen::MatrixXd> transitions, Eigen::MatrixXd rewards, double gamma):
		_states(states), _actions(actions), _transitions(transitions), 
		_rewards(rewards), _gamma(gamma) { /*Do Nothing*/ }

	Eigen::MatrixXd value_iteration() {
		double error = 1;
		Eigen::MatrixXd Jcurr = Eigen::MatrixXd::Zero(_states.size(), 1);

		Eigen::MatrixXd ca;
		Eigen::MatrixXd Jprev;
		Eigen::MatrixXd Pa;
		Eigen::MatrixXd Qmax;
		Eigen::MatrixXd Qa;

		while(error > 1e-8) {
			for(int i=0; i<_actions.size(); i++) {
				ca = _rewards.col(i);
				Pa = _transitions[i];
				Qa = ca + _gamma * Pa * Jcurr;
				if(i==0) { Qmax = Qa; }
				else     { Qmax = Qmax.array().max(Qa.array()); }  
			}

			Jprev = Jcurr;
			Jcurr = Qmax;
			error = (Jcurr - Jprev).norm();	
		}

		std::cout << "Jcurr -> " << Jcurr.rows() << " x " << Jcurr.cols() << std::endl;
		std::cout << "Jprev -> " << Jprev.rows() << " x " << Jprev.cols() << std::endl;
		std::cout << "Qa    -> " << Qa.rows() << " x " << Qa.cols() << std::endl;
		std::cout << "ca    -> " << ca.rows() << " x " << ca.cols() << std::endl;
		std::cout << "Qmax  -> " << Qmax.rows() << " x " << Qmax.cols() << std::endl;
		std::cout << "Pa    -> " << Pa.rows() << " x " << Pa.cols() << std::endl;
		return compute_policy_from_J(Jcurr);
	}

	Eigen::MatrixXd compute_policy_from_J(Eigen::MatrixXd J) {
		Eigen::MatrixXd pi(_states.size(), _actions.size());
		Eigen::MatrixXd  Q(_states.size(), _actions.size());

		for(int i=0; i<_actions.size(); i++)
			Q.col(i) = _rewards.col(i) + _gamma * _transitions[i] * J;

		pi.fill(0);
		int coef = 0;
		for(int i=0; i<_states.size(); i++) {
			for(int j=0; j<_actions.size(); j++) {
				if(std::fabs(Q(i, j)-J(i))<1e-5) {
					pi(i, j) = 1;
					coef++;
				}
			}
			pi.row(i) /= coef;
			coef = 0;
		}
		return pi;
	} 

	void follow_policy(Eigen::MatrixXd policy, int steps, bool finish_when_in_goal=false) {
		int curr_state = determine_state_of_robot();
		int next_state = 0;

		move_base_msgs::MoveBaseGoal goal;
		for(int step=0; step<steps; step++) {
			actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> ac("move_base", true);

			while(!ac.waitForServer(ros::Duration(5.0))){
				ROS_INFO("Waiting for the move_base action server to come up");
			}
			
			next_state = determine_new_state(curr_state, policy);

			goal.target_pose.header.frame_id = "map";
			goal.target_pose.header.stamp = ros::Time::now();

			goal.target_pose.pose.position.x = _states[next_state].first;
			goal.target_pose.pose.position.y = _states[next_state].second;
			goal.target_pose.pose.orientation.w = 0.2;
			goal.target_pose.pose.orientation.z = 0.0;

			ROS_INFO("Sending goal to reach state %d in (%.2f, %.2f)", next_state,
			         _states[next_state].first,
					 _states[next_state].second);
					
			ac.sendGoal(goal);
			ac.waitForResult();

			if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
				ROS_INFO("Hooray, the robot move to state %d in (%.2f, %.2f)", next_state,
				         _states[next_state].first,
				         _states[next_state].second);
			else
				ROS_INFO("The robot failed to reach state %d in (%.2f, %.2f)", next_state,
				         _states[next_state].first,
				         _states[next_state].second);

			curr_state = next_state;

			if(_rewards(curr_state, 0) == GOAL && finish_when_in_goal) {
				return;
			}
		}
	}

	int determine_state_of_robot() {
		Eigen::MatrixXd vector_pose(2, 1);
		vector_pose << __x, __y;

		std::cout << "Robot in " << __x << " " << __y << std::endl;

		int state = 0;
		std::pair<float, float> pos(-1, -1);
		double min_norm = INFINITY;
		for(int i=0; i<_states.size(); i++) {
			Eigen::MatrixXd vector_state(2, 1);
			vector_state << _states[i].first, _states[i].second;

			if( (vector_pose - vector_state).norm() < min_norm ) {
				min_norm = (vector_pose-vector_state).norm();
				state = i;
				pos = _states[i];
			} 
		}

		std::cout << "Curr state -> " << state;
		std::cout << " -- " << pos.first << " x " << pos.second << std::endl;


		return state;
	}

	int determine_new_state(int curr_state, Eigen::MatrixXd policy) {
		int action    = random_choice(policy.row(curr_state));
		int new_state = random_choice(_transitions[action].row(curr_state));

		return new_state;
	}

	int random_choice(Eigen::MatrixXd distribution) {
		double prob = ((double)((std::rand() % 100)+1)) / 100;
		double sum  = 0;

		for(int i=0; i<distribution.cols(); i++) {
			sum += distribution(i);
			if(sum >= prob) { return i; }
		}
	}

};

/*===================================================================
| map_discretizer -> the intent of this class is to encapsulate
| functions that allow one to take an OccupancyGrid from a ROS
| map_server and do everything in orther to discretize the map and
| turn it into an mdp
===================================================================*/
class map_discretizer {
	/* The grid map we got from the map_server */
	nav_msgs::OccupancyGrid _map;

	/* Lowest x with a pixel that is free */
	int _init_x;

	/* Lowest y with a pixel that is free */
	int _init_y;

	/* Highest x with a pixel that is free */
	int _final_x;

	/* Highest y with a pixel that is free */
	int _final_y;

	/* Side of each resulting cell in meters*/
	double _cell_size;

	/* 
	| Only used to print out the maps - how big do you want 
	| you resulting map images? Has to be larger than 1
	 */
	int _scale;

	/* 
	| Saves the color of each cell after discretization
	| The color also works as a code to see occupancy
	*/
	char** _square_colors;

	/* The color of the pixels in the resulting maps */
	char** _pixel_colors;

	/*===============================================================
	| compute the lowest x with a pixel that is free
	===============================================================*/
	void compute_init_x() {
		for(int x=0; x<_map.info.width; x++)
			for(int y=0; y<_map.info.height; y++)
				if (_map.data[y*_map.info.height+x] == 0) {
					_init_x  = x;
					return;
				}
	}

	/*===============================================================
	| compute the lowest y with a pixel that is free
	===============================================================*/
	void compute_init_y() {
		for(int y=0; y<_map.info.height; y++) 
			for(int x=0; x<_map.info.width; x++) 
				if (_map.data[y*_map.info.height+x] == 0) {
					_init_y = y;
					return;
				}
	}

	/*===============================================================
	| compute the highest x with a pixel that is free
	===============================================================*/
	void compute_final_x() {
		for(int x=_map.info.width-1; x>=0; x--)
			for(int y=0; y<_map.info.height; y++)
				if (_map.data[y*_map.info.height+x] == 0) {
					_final_x  = x;
					return;
				}
	}

	/*===============================================================
	| compute the highest y with a pixel that is free
	===============================================================*/
	void compute_final_y() {
		for(int y=_map.info.height-1; y>=0; y--)
			for(int x=0; x<_map.info.width; x++)
				if (_map.data[y*_map.info.height+x] == 0) {
					_final_y = y;
					return;
				}
	}

	/*===============================================================
	| check if a point (x, y) is within the bounds where we know 
	| for sure there is only free and occupied space - not unknown
	===============================================================*/
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

	/*===============================================================
	| The real word x coordinate of a pixel with coordinate x in 
	| the grid map
	===============================================================*/
	double pixel_x_to_coordinate(int x) {
		return x*_map.info.resolution + _map.info.origin.position.x;
	}

	/*===============================================================
	| The real word y coordinate of a pixel with coordinate y in 
	| the grid map
	===============================================================*/
	double pixel_y_to_coordinate(int y) {
		return y*_map.info.resolution + _map.info.origin.position.y;
	}

	/*===============================================================
	| The resulting width of the map in cells after discretization
	===============================================================*/
	int discretized_grid_size_x() {
		return (_final_x - _init_x)*_map.info.resolution/_cell_size + 1;
	}

	/*===============================================================
	| The resulting height of the map in cells after discretization
	===============================================================*/
	int discretized_grid_size_y() {
		return (_final_y - _init_y)*_map.info.resolution/_cell_size + 1;
	}

	/*===============================================================
	| The resulting number of totally free cells in the resulting
	| discretization
	===============================================================*/
	int num_free_squares() {
		int res = 0;
		determine_square_colors();
		for(int y=discretized_grid_size_y()-1; y>=0; y--)
			for(int x=0; x<discretized_grid_size_x(); x++)
				if(_square_colors[x][y] != BLACK)
					res++;
		return res;
	}

	/*===============================================================
	| Print to screen the central coordinate of each cell if it is 
	| fully free or OCCUPIED if the cell will not be considered .i.e
	| it's not totally free
	===============================================================*/
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

	/*===============================================================
	| Paint a picture with the received map, scaled with _scale
	===============================================================*/
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

	/*===============================================================
	| Paint a picture with the discretized map, scaled with _scale,
	| at this point, it will not leave slightly occupied squares
	| out of the picture
	===============================================================*/
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
								img << 52 << " " << 143 << " " << 235 << std::endl;
							else
								img << 52 << " " << 85 << " " << 235 << std::endl;
						}
						else {
							if( (((int)(y-_init_y))%((int) (2*_cell_size/_map.info.resolution + 1))) < _cell_size/_map.info.resolution )
								img << 52 << " " << 143 << " " << 235 << std::endl;
							else
								img << 52 << " " << 85 << " " << 235 << std::endl;
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

	/*===============================================================
	| Compute color of each square - essecially discretized
	===============================================================*/
	void determine_square_colors() {
		for(int y=discretized_grid_size_y()-1; y>=0; y--)
			for(int x=0; x<discretized_grid_size_x(); x++)
				_square_colors[x][y] = determine_color_of_square(x, y);
	}

	/*===============================================================
	| Compute color of each resulting pixel, has to be called
	| after computing the square colors (determine_square_colors)
	===============================================================*/
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
						_pixel_colors[x][y] = BLUE1;
					}
					else if(x_square%2==0 && y_square%2!=0) {
						_pixel_colors[x][y] = BLUE2;
					}
					else if(x_square%2!=0 && y_square%2!=0) {
						_pixel_colors[x][y] = BLUE1;
					}
					else if(x_square%2!=0 && y_square%2==0) {
						_pixel_colors[x][y] = BLUE2;
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

	/*===============================================================
	| Compute color of each resulting pixel, accounting for the 
	| respective rewards in each specific square has to be called
	| after computing the square colors (determine_square_colors)
	===============================================================*/
	void determine_pixel_rewards_colors(Eigen::MatrixXd rewards) {
		for(int x=0; x<_map.info.width; x++) {
			for(int y=0; y<_map.info.height; y++) {
				if(point_within_bounds(x, y)) {
					auto square = determine_square_of_pixel(x, y);
					int x_square = square.first;
					int y_square = square.second;
					if(_square_colors[x_square][y_square]==BLACK) {
						_pixel_colors[x][y] = BLACK;
					}
					else if(rewards(get_index_of_state_square(x_square, y_square), 0) == DEADLY) {
						if(x_square%2==0 && y_square%2==0) {
							_pixel_colors[x][y] = RED1;
						}
						else if(x_square%2==0 && y_square%2!=0) {
							_pixel_colors[x][y] = RED2;
						}
						else if(x_square%2!=0 && y_square%2!=0) {
							_pixel_colors[x][y] = RED1;
						}
						else if(x_square%2!=0 && y_square%2==0) {
							_pixel_colors[x][y] = RED2;
						}
					}
					else if(rewards(get_index_of_state_square(x_square, y_square), 0) == NORMAL) {
						if(x_square%2==0 && y_square%2==0) {
							_pixel_colors[x][y] = BLUE1;
						}
						else if(x_square%2==0 && y_square%2!=0) {
							_pixel_colors[x][y] = BLUE2;
						}
						else if(x_square%2!=0 && y_square%2!=0) {
							_pixel_colors[x][y] = BLUE1;
						}
						else if(x_square%2!=0 && y_square%2==0) {
							_pixel_colors[x][y] = BLUE2;
						}
					}
					else if(rewards(get_index_of_state_square(x_square, y_square), 0) == GOAL) {
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

	/*===============================================================
	| Compute to which discreteized cell the pixel (x, y) corresponds
	| to - returns the coordinates in the discretized grid
	===============================================================*/
	std::pair<int, int> determine_square_of_pixel(int x, int y) {
		int x_delocation = x-_init_x;
		int y_delocation = y-_init_y;
		int x_square = x_delocation/(((double)_cell_size/_map.info.resolution));
		int y_square = y_delocation/(((double)_cell_size/_map.info.resolution));
		std::pair<int, int> res(x_square, y_square);
		return res;
	}

	/*===============================================================
	| Compute color of square (x,y) in the resulting discretized grid
	===============================================================*/
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

	/*===============================================================
	| Paint a picture with the discretized map, scaled with _scale,
	| at this point cells that are slightly occupied in the real 
	| world will be painted all black - the colored squares will
	| be the only cells considered to build the mdp
	===============================================================*/
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
				else if(_pixel_colors[x][y] == BLUE1) {
					img << 52 << " " << 143 << " " << 235 << std::endl;
				}
				else if(_pixel_colors[x][y] == BLUE2) {
					img << 52 << " " << 85 << " " << 235 << std::endl;
				}
			}
		}
	}

	/*===============================================================
	| Paint a picture with the discretized map, scaled with _scale,
	| at this point cells that are slightly occupied in the real 
	| world will be painted all black - the colored squares will
	| be the only cells considered to build the mdp and squares will
	| be color coded to represent the reward of executing an action
	| in that specific state
	===============================================================*/
	void paint_chess_filtered_reward_map(const std::string& path, Eigen::MatrixXd rewards) {
		determine_square_colors();
		determine_pixel_rewards_colors(rewards);

		std::cout << "Where was it?" << std::endl;

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
				else if(_pixel_colors[x][y] == RED1) {
					img << 235 << " " << 52 << " " << 52 << std::endl;
				}
				else if(_pixel_colors[x][y] == RED2) {
					img << 255 << " " << 110 << " " << 110 << std::endl;
				}
				else if(_pixel_colors[x][y] == BLUE1) {
					img << 52 << " " << 143 << " " << 235 << std::endl;
				}
				else if(_pixel_colors[x][y] == BLUE2) {
					img << 52 << " " << 85 << " " << 235 << std::endl;
				}
				else if(_pixel_colors[x][y] == GREEN) {
					img << 117 << " " << 255 << " " << 168 << std::endl;
				}
			}
		}
	}

	/*===============================================================
	| Cumpute states for the MDP - each state is just a point in the
	| real world - (x, y)
	===============================================================*/
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

	/*===============================================================
	| Compute transition probabilities for the MDP
	| The robot will move up, down, right and left - each action
	| as probability SUCC of succeeding, when it fails it will
	| mode to one of the adjacent states or stay in the present cell
	===============================================================*/
	std::vector<Eigen::MatrixXd> build_transitions_for_mdp() {
		std::vector<Eigen::MatrixXd> transitions;
		transitions.push_back(build_up_transition());
		transitions.push_back(build_down_transition());
		transitions.push_back(build_right_transition());
		transitions.push_back(build_left_transition());
		return transitions;	
	}

	/*===============================================================
	| Compute the state number that the square (x, y) in the 
	| discretized grid corresponds to
	===============================================================*/
	int get_index_of_state_square(int x, int y) {
		int res = 0;
		for(int yi=discretized_grid_size_y()-1; yi>=0; yi--) {
			for(int xi=0; xi<discretized_grid_size_x(); xi++) { 
				if(xi==x && yi==y) { return res; }
				if(_square_colors[xi][yi] != BLACK) { res++; } 
			}
		}
	}

	/*===============================================================
	| Compute the state number that the center point (x, y)
	| represents - remember that each cell is characterized by
	| its real world center coordinate
	===============================================================*/
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

	/*===============================================================
	| Compute transitions probabilities for the UP action
	===============================================================*/
	Eigen::MatrixXd build_up_transition() {
		int squares = num_free_squares();
		Eigen::MatrixXd up(squares, squares);
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
				up(get_index_of_state_square(x, y), get_index_of_state_square(x, y)) = sum;
				sum = 0;
				
			}
		}
		return up;
	}

	/*===============================================================
	| Compute transitions probabilities for the DOWN action
	===============================================================*/
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
				down(get_index_of_state_square(x, y), get_index_of_state_square(x, y)) = sum;
				sum = 0;
			}
		}
		return down;	
	}

	/*===============================================================
	| Compute transitions probabilities for the RIGHT action
	===============================================================*/
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
				right(get_index_of_state_square(x, y), get_index_of_state_square(x, y)) = sum;
				sum = 0;
			}
		}
		return right;
	}

	/*===============================================================
	| Compute transitions probabilities for the LEFT action
	===============================================================*/
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
				left(get_index_of_state_square(x, y), get_index_of_state_square(x, y)) = sum;
				sum = 0;
			}
		}
		return left;
	}

	/*===============================================================
	| Compute reward function for the MDP - receives the goal (which
	| will have higher reward) and the cells to avoid (which will 
	| have very low rewards), all the oders have the same rewards
	| This function awards the same for every action in each state
	===============================================================*/
	Eigen::MatrixXd build_rewards_for_mdp(std::pair<float, float> goal, std::vector<std::pair<float, float>> to_avoid) {
		int squares = num_free_squares();
		Eigen::MatrixXd rewards(squares, 4);
		rewards.fill(NORMAL);

		for(auto point : to_avoid) {
			int index =  get_index_of_state_by_coord(point.first, point.second);
			rewards(index, 0) = DEADLY;
			rewards(index, 1) = DEADLY;
			rewards(index, 2) = DEADLY;
			rewards(index, 3) = DEADLY;
		}
		
		int goal_index = get_index_of_state_by_coord(goal.first, goal.second);
		rewards(goal_index, 0) = GOAL;
		rewards(goal_index, 1) = GOAL;
		rewards(goal_index, 2) = GOAL;
		rewards(goal_index, 3) = GOAL;

		return rewards;
	}

};

int main(int argc, char* argv[]) {
	ros::init(argc, argv, "map_discretizer");
	ros::init(argc, argv, "markov_goals");
	ros::init(argc, argv, "pose_getter");

	ros::NodeHandle n;

	ros::Subscriber sub = n.subscribe("amcl_pose", 100, chatterCallback);

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

	// m_d.paint_chess_filtered_map("filtered_map.ppm");
	// m_d.paint_chess_map("squared_map.ppm");
	// m_d.paint_map("normal_map.ppm");
	// m_d.print_position_of_pixel(100, 100);
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
	double gamma = 0.9;

	m_d.paint_chess_filtered_reward_map("rewards_map.ppm", rewards);

	/*PROOF THAT THE VALUE ITERATION WORKS*/
	// std::vector<std::pair<float, float>> states = {std::pair<float, float>(1, 1),
	//                                                std::pair<float, float>(1, 1),
	// 											   std::pair<float, float>(1, 1) };
	// Eigen::MatrixXd m1(3, 3);
	// m1 << 0, 1, 0, 0, 1, 0, 0, 0, 1;
	// Eigen::MatrixXd m2(3, 3);
	// m2 << 0, 0, 1, 0, 1, 0, 0, 0, 1;
	// std::vector<Eigen::MatrixXd> transitions = {m1, m2};
	// Eigen::MatrixXd rewards(3, 2);
	// rewards << 1, 0.5, 0, 0, 1, 1;
	// std::vector<std::string> actions = {"a", "b"};
	// double gamma = 0.9;

	mdp problem(states, actions, transitions, rewards, gamma);
	Eigen::MatrixXd policy = problem.value_iteration();

	std::cout << policy << std::endl;

	//problem.follow_policy(policy, 1000);

	/*END OF PROOF THAT THE VALUE ITERATION WORKS*/


	// std::cout << rewards << std::endl;

	// for(auto m : transitions) {
	// 	std::cout << m << std::endl;
	// 	std::cout << m.rows() << " " << m.cols() << std::endl;
	// }
	
	// std::cout << states.size() << std::endl;

	// int i = 0;
	// for(auto p : states) {
	// 	std::cout << "State -> " << i++ << " <-> " << p.first << " -- " << p.second << std::endl;
	// }

	ROS_INFO("Finished");

	return 0;
}